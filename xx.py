import torch

from einops import rearrange


# key = torch.as_tensor(list(range(16)))
key = torch.as_tensor([0,5,0,5])

chunk_size = 2
video_length = key.size()[0] // chunk_size
print(video_length)

former_frame_index = [0] * video_length
print(key)
key = rearrange(key, "(b f) -> b f", f=video_length)
print(key)
key = key[:, former_frame_index]
print(key)
key = rearrange(key, "b f -> (b f)")
print(key)

