
#### a demo
# # Define the arguments you want to pass to the Python script
# args=(
#     "configs/xinye2-Scene-001.yaml" 
#     "configs/xinye2-Scene-001.yaml" 
#     "configs/xinye2-Scene-001.yaml"
# )

# cd /home/zhaizhichao/gkf_proj/Inpaint-Anything/
# # Loop through each argument and run the Python script
# for arg in "${args[@]}"; do
#     CUDA_VISIBLE_DEVICES=0 python args_demo.py \
#         --yaml_cfg_path "$arg"
# done


args=(
    "configs/xinye1-Scene-001-wedding.yaml" 
    # "configs/xinye2-Scene-001.yaml" 
    # "configs/xinye3-Scene-001.yaml"
    # "configs/xinye4-Scene-001.yaml"
    # "configs/xinye5-Scene-001.yaml"
)

cd /home/zhaizhichao/gkf_proj/Inpaint-Anything/
# Loop through each argument and run the Python script
for arg in "${args[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python replace_anything_video.py \
        --yaml_cfg_path "$arg"
done

# CUDA_VISIBLE_DEVICES=1 python replace_anything_video.py --yaml_cfg_path configs/xinye1-Scene-001.yaml