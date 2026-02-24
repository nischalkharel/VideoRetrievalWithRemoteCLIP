# Author: Nischal Kharel
# Date: 2/24/2026
# https://github.com/ChenDelong1999/RemoteCLIP?tab=readme-ov-file

import torch
import open_clip
from PIL import Image

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

text_queries = [
    "A busy airport with many airplanes.", 
    "Satellite view of Hohai University.", 
    "A building next to a lake.", 
    "Many people in a stadium.", 
    "a cute cat",
    ]

text = tokenizer(text_queries)
image = preprocess(Image.open("data/ss.jpg")).unsqueeze(0)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]

print(f'Predictions of {model_name}:')
for query, prob in zip(text_queries, text_probs):
    print(f"{query:<40} {prob * 100:5.1f}%")