# Author: Nischal Kharel
# Date: 2/24/2026

from huggingface_hub import hf_hub_download
import os

#Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

#Download ViT-L-14 (change it to RemoteCLIP-ViT-B-32.pt if we want to download the smaller model)
model_variant = "RemoteCLIP-ViT-L-14.pt" #https://huggingface.co/chendelong/RemoteCLIP/tree/main 
checkpoint_path = hf_hub_download(
            repo_id = "chendelong/RemoteCLIP",
            filename = model_variant,
            local_dir = "models", #Download the model into models directory
)

print(f'Model downloaded and saved to: {checkpoint_path}')