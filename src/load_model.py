# Author: Nischal Kharel
# Date: 2/24/2026

import torch
import open_clip

#build base structure
model_name = "ViT-L-14" # Change to "ViT-B-32" if you want to load the smaller model
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

# Load the downloaded checkpoint
checkpoint_path = "models/RemoteCLIP-ViT-L-14.pt" # Change to "models/RemoteCLIP-ViT-B-32.pt" if using the smaller model
checkpoint = torch.load(checkpoint_path, map_location="cpu")
message = model.load_state_dict(checkpoint)
print(message)
model.eval() # Set the model to evaluation mode

print("model loaded successfully and ready for inference!")