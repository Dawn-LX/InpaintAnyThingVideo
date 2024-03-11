import sys
import argparse
from omegaconf import OmegaConf

from pathlib import Path

default_cfgs = dict(
    input_imgs_dir = "",
    coords_type = "key_in",
    point_coords = [0.5,0.5],
    point_labels = 1,
    text_prompt = "",
    dilate_kernel_size = 5,
    output_dir = "/home/zhaizhichao/gkf_proj/Inpaint-Anything/results2",
    sam_model_type = "vit_h",
    sam_ckpt  = "/home/zhaizhichao/gkf_proj/Inpaint-Anything/pretrained_models/sam_vit_h_4b8939.pth",
    seed = 119,
    sample_rate = 4,
    use_cf_attn = True,
)
default_cfgs = OmegaConf.create(default_cfgs)

parser = argparse.ArgumentParser()
parser.add_argument("--yaml_cfg_path", type=str,help="yaml_cfg_path")
args = parser.parse_args(sys.argv[1:])

if args.yaml_cfg_path is not None:
    extra_cfgs = OmegaConf.load(args.yaml_cfg_path)
    print(args.yaml_cfg_path)
    default_cfgs.update(extra_cfgs)
    print(default_cfgs)

import os
for name in os.listdir("configs"):
    print(name)

# input_imgs_dir  = "/home/zhaizhichao/gkf_proj/FateZero-diffusers-0.11.1/FateZero-main/data/xinye2-Scene-001.mp4.all_frames.crop"
# img_stem = Path(input_imgs_dir).name
# # out_dir = os.path.join(args.output_dir,str(img_stem))
# print(str(img_stem))