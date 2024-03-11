import cv2
import os
import sys
import argparse
import random
from omegaconf import OmegaConf
from collections import defaultdict
import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from typing import Any, Dict, List
from diffusers import StableDiffusionInpaintPipeline
from sam_segment import predict_masks_with_sam
from utils.crop_for_replacing import recover_size, resize_and_pad
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point,save_images_as_mp4

IS_DEBUG = os.getenv("IS_DEBUG","0").lower() in ["true","yes","1"]

from sd_inpaint_video_pipeline import SDInpaintVideoPipeline

class CrossFrameAttnProcessor:
    def __init__(self, unet_chunk_size=2):
        self.unet_chunk_size = unet_chunk_size
        print("args.use_cf_attn = True, CrossFrameAttnProcessor init finish")
    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        # elif attn.cross_attention_norm: # Text2Video-Zero uses diffusers==0.14.0
        # elif attn.norm_cross is not None: # here we use diffusers==0.23.1
        elif hasattr(attn,"norm_cross") and (attn.norm_cross is not None): # for both diffusers==0.14.0 and diffusers==0.23.1
            '''gkf: NOTE
                for diffusers==0.14.0: 
                    attn.cross_attention_norm is `True` or `False`, and attn.norm_cross is `nn.LayerNorm`
                refer to https://github.com/huggingface/diffusers/blob/f20c8f5a1aba27f5972cad50516f18ba516e4d9e/src/diffusers/models/cross_attention.py#L69

                for diffusers==0.23.1: 
                    `cross_attention_norm` can be `None`, "layer_norm", or "group_norm"
                    , and `attn` does not store the `cross_attention_norm` as `self.cross_attention_norm`
                    , instead, it uses self.norm_cross, which can be `None`, `nn.LayerNorm`, or `nn.GroupNorm`
                refer to https://github.com/huggingface/diffusers/blob/4719b8f5f9714f1c0b3fd32addaf0f61a9939219/src/diffusers/models/attention_processor.py#L153
            '''
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # Sparse Attention
        if not is_cross_attention:
            video_length = key.size()[0] // self.unet_chunk_size
            # former_frame_index = torch.arange(video_length) - 1
            # former_frame_index[0] = 0
            former_frame_index = [0] * video_length
            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            key = key[:, former_frame_index]
            key = rearrange(key, "b f d c -> (b f) d c")
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            value = value[:, former_frame_index]
            value = rearrange(value, "b f d c -> (b f) d c")

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

def setup_args(parser):
    parser.add_argument(
        "--input_imgs_dir", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--coords_type", type=str, required=True,
        default="key_in", choices=["click", "key_in"], 
        help="The way to select coords",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', required=True,
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+', required=True,
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--text_prompt", type=str, required=True,
        help="Text prompt",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
        help="The type of sam model to load. Default: 'vit_h"
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--seed", type=int,default=111,
        help="Specify seed for reproducibility.",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use deterministic algorithms for reproducibility.",
    )

    parser.add_argument(
        "--sample_rate", type=int, default=None,
        help="sample_rate. Default: None",
    )
    parser.add_argument(
        "--use_cf_attn", action='store_true',
        help="The path to the SAM checkpoint to use for mask generation.",
    )


def coord_norm2abs(coord,orig_wh):
    w,h = orig_wh[0],orig_wh[1]
    x,y = coord[0],coord[1]

    y_abs = y*h
    x_abs = x*w

    return [x_abs,y_abs]

def replace_img_with_sd(
        pipe,
        img: np.ndarray,
        mask: np.ndarray,
        text_prompt: str,
        step: int = 50,
        seed = None,
):
    if seed is not None:
        generator = torch.Generator(device='cuda').manual_seed(seed)

    img_padded, mask_padded, padding_factors = resize_and_pad(img, mask)
    img_padded = pipe(
        prompt=text_prompt,
        image=Image.fromarray(img_padded),
        mask_image=Image.fromarray(255 - mask_padded),
        num_inference_steps=step,
        generator=generator,
    ).images[0]
    # print(type(img_padded),"after pipe forward")
    height, width, _ = img.shape
    img_resized, mask_resized = recover_size(
        np.array(img_padded), mask_padded, (height, width), padding_factors)
    mask_resized = np.expand_dims(mask_resized, -1) / 255
    img_resized = img_resized * (1-mask_resized) + img * mask_resized
    return img_resized


def save_segmentation_results(args,img,latest_coords,masks,out_dir):
    # visualize the segmentation results
    
    for idx, mask in enumerate(masks):
        # path to the results
        mask_p = os.path.join(out_dir,f"mask_{idx}.png")
        img_points_p = os.path.join(out_dir,f"with_points.png")
        img_mask_p =  os.path.join(out_dir,f"with_{Path(mask_p).name}")

        # save the mask
        save_array_to_img(mask, mask_p)

        # save the pointed and masked image
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), [latest_coords], args.point_labels,
                    size=(width*0.04)**2)
        plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()

def sample_frames(img_paths,sample_rate,max_L=64):
    img_paths_new = [p for i,p in enumerate(img_paths) if i%sample_rate==0]
    img_paths_new = img_paths_new[:max_L]
    print("sampled imgs:",img_paths_new[:4],"...",img_paths_new[-1],f"with sample_rate={sample_rate}")

    return img_paths_new


def main(args):


    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.coords_type == "click":
        latest_coords = get_clicked_point(args.input_img)
    elif args.coords_type == "key_in":
        latest_coords = args.point_coords
    
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
    ).to(device)
    # /home/zhaizhichao/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-inpainting/snapshots/81a84f49b15956b60b4272a405ad3daef3da4590

    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.use_cf_attn:
        cross_frame_attn_proc = CrossFrameAttnProcessor(unet_chunk_size=2)
        pipe.unet.set_attn_processor(processor=cross_frame_attn_proc)
        # assert False, "refer to /home/zhaizhichao/gkf_proj/Text2Video-Zero/sd_inpaint_anything_in_video.py"
        
    
    ######
    max_L=64
    results_seq = defaultdict(list)
    img_paths = [os.path.join(args.input_imgs_dir,name) for name in sorted(os.listdir(args.input_imgs_dir))]
    if args.sample_rate is not None:
        img_paths = sample_frames(img_paths,args.sample_rate,max_L)
    if IS_DEBUG: img_paths = img_paths[:16]

    img_paths = img_paths[:max_L]
    if latest_coords[0]<1 and latest_coords[1] < 1:
        _frame0 = load_img_to_array(img_paths[0])
        h,w,_ = _frame0.shape
        orig_wh = (w,h)
        latest_coords = coord_norm2abs(latest_coords,orig_wh)
    
    total = len(img_paths)
    pipe.set_progress_bar_config(disable=True)
    for frame_id,img_path in enumerate(tqdm(img_paths,desc="loop for img seq")):

        img = load_img_to_array(img_path)
    
    
        masks, _, _ = predict_masks_with_sam(
            img,
            [latest_coords],
            args.point_labels,
            model_type=args.sam_model_type,
            ckpt_p=args.sam_ckpt,
            device=device,
        )
        masks = masks.astype(np.uint8) * 255


        # dilate mask to avoid unmasked edge effect
        if args.dilate_kernel_size is not None:
            masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

        if frame_id==0:
            save_segmentation_results(args,img,latest_coords,masks,args.output_dir)
    
        if IS_DEBUG: print(img.shape,masks.shape)
        # fill the masked image
        for idx, mask in enumerate(masks):
            if idx!=2: continue
            # mask_p = out_dir / f"mask_{idx}.png"
            # img_replaced_p = out_dir / f"replaced_with_{Path(mask_p).name}"
            img_replaced = replace_img_with_sd(
                pipe,
                img, 
                mask, 
                args.text_prompt,
                seed=args.seed
            )
            img_replaced = Image.fromarray(img_replaced.astype(np.uint8))
            results_seq[idx].append(img_replaced)
        # print(f"frame_id={frame_id}, total: {frame_id}/{total}")
            
    img_seq = results_seq[2]



    sr = args.sample_rate if args.sample_rate is not None else 1

    output_video_path = os.path.join(args.output_dir,f"out_video.sr{sr}.seed{args.seed}.mp4")
    if IS_DEBUG: output_video_path = output_video_path+".debug.mp4"

    save_images_as_mp4(img_seq,output_video_path)
    print(f"saved at {output_video_path}")


def main_cf_attn(args):


    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.coords_type == "click":
        latest_coords = get_clicked_point(args.input_img)
    elif args.coords_type == "key_in":
        latest_coords = args.point_coords
    
    
    pipe = SDInpaintVideoPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
    ).to(device)
    # /home/zhaizhichao/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-inpainting/snapshots/81a84f49b15956b60b4272a405ad3daef3da4590

    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.use_cf_attn:
        cross_frame_attn_proc = CrossFrameAttnProcessor(unet_chunk_size=2)
        pipe.unet.set_attn_processor(processor=cross_frame_attn_proc)
        # assert False, "refer to /home/zhaizhichao/gkf_proj/Text2Video-Zero/sd_inpaint_anything_in_video.py"
        
    
    ######
    max_L=64
    results_seq = defaultdict(list)
    img_paths = [os.path.join(args.input_imgs_dir,name) for name in sorted(os.listdir(args.input_imgs_dir))]
    if args.sample_rate is not None:
        img_paths = sample_frames(img_paths,args.sample_rate,max_L)
    if IS_DEBUG: img_paths = img_paths[:4]

    img_paths = img_paths[:max_L]
    if latest_coords[0]<1 and latest_coords[1] < 1:
        _frame0 = load_img_to_array(img_paths[0])
        h,w,_ = _frame0.shape
        orig_wh = (w,h)
        latest_coords = coord_norm2abs(latest_coords,orig_wh)
    
    sr = args.sample_rate if args.sample_rate is not None else 1
    x,y = latest_coords
    output_dir = os.path.join(args.output_dir,f"x{int(x)}_y{int(y)}.sr{sr}")
    os.makedirs(output_dir,exist_ok=True)
    output_video_path = os.path.join(output_dir,f"seed{args.seed}_{args.exp_tag}.mp4")
    mask_cache_path = os.path.join(output_dir,"masks3_cache.npy")
    if IS_DEBUG: mask_cache_path = mask_cache_path+".debug.npy"

    total = len(img_paths)
    pipe.set_progress_bar_config(disable=True)
    

    if use_mask_cache := os.path.exists(mask_cache_path):
        print("load mask cache from",mask_cache_path)
        cached_masks_before_dilate = np.load(mask_cache_path)
        cached_masks_before_dilate = [cached_masks_before_dilate[frame_id] for frame_id in range(cached_masks_before_dilate.shape[0])]
    else:
        print("no mask cache found, run SAM and it will be saved at",mask_cache_path)
        cached_masks_before_dilate = []
    
    img_seq,mask_seq = [],[]
    for frame_id,img_path in enumerate(tqdm(img_paths,desc="apply SAM to each frame")):

        img = load_img_to_array(img_path)
        img_seq.append(img)

        if use_mask_cache:
            masks = cached_masks_before_dilate[frame_id]
        else:
            masks, _, _ = predict_masks_with_sam(
                img,
                [latest_coords],
                args.point_labels,
                model_type=args.sam_model_type,
                ckpt_p=args.sam_ckpt,
                device=device,
            )
            masks = masks.astype(np.uint8) * 255
            cached_masks_before_dilate.append(masks)
        

        # dilate mask to avoid unmasked edge effect
        if args.dilate_kernel_size is not None:
            masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

        if frame_id==0:
            save_segmentation_results(args,img,latest_coords,masks,output_dir)
    
        if IS_DEBUG: print(img.shape,masks.shape)
        mask_seq.append(masks[2])
    
    if not os.path.exists(mask_cache_path):
        np.save(mask_cache_path,cached_masks_before_dilate)
    
    img_replaced_seq = []
    for frame_id in tqdm(range(len(img_seq)),desc="pipeline froward"):
        generator = torch.Generator(device='cuda').manual_seed(args.seed)

        img = img_seq[frame_id]
        mask = mask_seq[frame_id]

        img_padded, mask_padded, padding_factors = resize_and_pad(img, mask)
        frame0,mask0,_ = resize_and_pad(img_seq[0], mask_seq[0])

        img_padded = pipe(
            prompt=[args.text_prompt]*2,
            image=[Image.fromarray(frame0),Image.fromarray(img_padded)],
            mask_image=[Image.fromarray(255 - mask0),Image.fromarray(255 - mask_padded)],
            num_inference_steps=50,
            generator=generator,
        )#.images[0] # list of PIL Image
        img_padded = img_padded[-1]  #
        if IS_DEBUG: print(type(img_padded),"after pipe forward")
        height, width, _ = img.shape
        img_resized, mask_resized = recover_size(
            np.array(img_padded), mask_padded, (height, width), padding_factors)
        mask_resized = np.expand_dims(mask_resized, -1) / 255
        img_resized = img_resized * (1-mask_resized) + img * mask_resized

        img_replaced_seq.append(Image.fromarray(img_resized.astype(np.uint8)))

    
    ###### save
    
    if IS_DEBUG: output_video_path = output_video_path+".debug.mp4"

    save_images_as_mp4(img_replaced_seq,output_video_path)
    print(f"saved at {output_video_path}")

def save_sampled_orig_video(args):
    max_L=64
    img_paths = [os.path.join(args.input_imgs_dir,name) for name in sorted(os.listdir(args.input_imgs_dir))]
    if args.sample_rate is not None:
        img_paths = sample_frames(img_paths,args.sample_rate,max_L)
    
    img_seq = [Image.open(img_path) for img_path in img_paths]
    sr = args.sample_rate if args.sample_rate is not None else 1
    output_video_path = os.path.join(args.output_dir,f"out_video.sr{sr}.orig.mp4")
    save_images_as_mp4(img_seq,output_video_path)
    print(output_video_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # setup_args(parser)
    parser.add_argument("--yaml_cfg_path", required=True, type=str,help="yaml_cfg_path")
    args = parser.parse_args(sys.argv[1:])

    default_cfgs = dict(
        input_imgs_dir = "TODO",
        coords_type = "key_in",
        point_coords = [0.5,0.5],
        point_labels = [1],
        text_prompt = "TODO",
        dilate_kernel_size = 5,
        output_dir = "/home/zhaizhichao/gkf_proj/Inpaint-Anything/results3",
        sam_model_type = "vit_h",
        sam_ckpt  = "/home/zhaizhichao/gkf_proj/Inpaint-Anything/pretrained_models/sam_vit_h_4b8939.pth",
        seed = None,
        sample_rate = 4,
        use_cf_attn = True,
        exp_tag = "", # extra str tag for diffenent exp try
    )
    configs = OmegaConf.create(default_cfgs)

    extra_cfgs = OmegaConf.load(args.yaml_cfg_path)
    configs.update(extra_cfgs)

    if not isinstance(configs.seed,int): #configs.seed is None:
        configs.seed = torch.randint(0,10000,size=()).item()

    img_stem = Path(configs.input_imgs_dir).stem
    output_dir = os.path.join(configs.output_dir,str(img_stem)+f".cf_attn_{configs.use_cf_attn}")
    os.makedirs(output_dir,exist_ok=True)
    configs.output_dir = output_dir
    
    print("configs:",configs)
    print("output_dir:",configs.output_dir,"seed:",configs.seed)

    
    # save_sampled_orig_video(configs)
    # if configs.use_cf_attn:
    main_cf_attn(configs)
    # else:
    #     main(configs)

    """Example usage:
    stands on the lawn in front of a river
    
    CUDA_VISIBLE_DEVICES=1 python replace_anything_video.py --yaml_cfg_path configs/xinye2-Scene-001.yaml

    ##### xinye3-Scene-001.mp4.all_frames (酒吧-红裙子)
    CUDA_VISIBLE_DEVICES=0 python replace_anything_video.py \
        --input_imgs_dir /home/zhaizhichao/gkf_proj/FateZero-diffusers-0.11.1/FateZero-main/data/xinye3-Scene-001.mp4.all_frames \
        --coords_type key_in \
        --point_coords 0.6 0.5 \
        --point_labels 1 \
        --text_prompt "sits in front of a wall with window" \
        --output_dir ./results2/ \
        --sam_model_type "vit_h" \
        --sam_ckpt /home/zhaizhichao/gkf_proj/Inpaint-Anything/pretrained_models/sam_vit_h_4b8939.pth \
        --dilate_kernel_size 5 \
        --seed 120 \
        --sample_rate 4 \
        --use_cf_attn \
    
    export HF_ENDPOINT=https://hf-mirror.com
    
    #### 坐着的 红衣服 results/
    CUDA_VISIBLE_DEVICES=0 python replace_anything_video.py \
        --input_imgs_dir /home/zhaizhichao/gkf_proj/FateZero-diffusers-0.11.1/FateZero-main/data/xinye2-Scene-001.mp4.all_frames.crop \
        --coords_type key_in \
        --point_coords 0.6 0.7 \
        --point_labels 1 \
        --text_prompt "sit in an office room" \
        --output_dir ./results2/ \
        --sam_model_type "vit_h" \
        --sam_ckpt /home/zhaizhichao/gkf_proj/Inpaint-Anything/pretrained_models/sam_vit_h_4b8939.pth \
        --dilate_kernel_size 5 \
        --seed 119 \
        --sample_rate 4 \
        --use_cf_attn \
    
    
    # 站着的黑色背景
    CUDA_VISIBLE_DEVICES=0 python replace_anything_video.py \
        --input_imgs_dir /home/zhaizhichao/gkf_proj/FateZero-diffusers-0.11.1/FateZero-main/data/xinye1-Scene-001.mp4.frames \
        --coords_type key_in \
        --point_coords 0.5 0.57 \
        --point_labels 1 \
        --text_prompt "stands on the lawn in front of a river" \
        --output_dir ./results2/ \
        --sam_model_type "vit_h" \
        --sam_ckpt /home/zhaizhichao/gkf_proj/Inpaint-Anything/pretrained_models/sam_vit_h_4b8939.pth \
        --dilate_kernel_size 5 \
        --seed 119 \
        --sample_rate 4 \
        --use_cf_attn \
    

    ### xinye4:  A woman stands on the lawn in front of a river, selfie view
        
    CUDA_VISIBLE_DEVICES=0 python replace_anything_video.py \
        --input_imgs_dir /home/zhaizhichao/gkf_proj/FateZero-diffusers-0.11.1/FateZero-main/data/xinye4-Scene-001.mp4.frames \
        --coords_type key_in \
        --point_coords 0.73 0.86 \
        --point_labels 1 \
        --text_prompt "stands on the lawn in front of a river" \
        --output_dir ./results2/ \
        --sam_model_type "vit_h" \
        --sam_ckpt /home/zhaizhichao/gkf_proj/Inpaint-Anything/pretrained_models/sam_vit_h_4b8939.pth \
        --dilate_kernel_size 5 \
        --seed 119 \
        --sample_rate 4 \
        --use_cf_attn \
    
    """