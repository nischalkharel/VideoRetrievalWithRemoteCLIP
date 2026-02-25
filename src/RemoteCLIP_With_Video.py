#This code is supposed to:
# 1. Load the video from .\data\video_name.mp4
# 2. Extract frames from the video at a specified frame rate
# 3. Preprocess the frames and encode them using the RemoteCLIP model to get image features
# 4. Tokenize the frames and store them in memory
# 5. Given a text query, encode the text using the RemoteCLIP model to get text features
# 6. Compute the cosine similarity between the text features and the image features to retrieve the most relevant frames
# 7. Display the retrieved frames along with their similarity scores

import torch
import open_clip
from PIL import Image
import cv2
import numpy as np
import os

#Loading the model first
model_name = "ViT-L-14"
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)
checkpoint_path = "models/RemoteCLIP-ViT-L-14.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
message = model.load_state_dict(checkpoint)
print(message)
model.eval()
print("model loaded successfully and ready for inference!")

embeddings_file = "data/short_video_embeddings.pt"

# Step 1-4: Load the video, extract frames, preprocess, encode them, store embeddings in memory
video_path = "data/short_video.mp4" # Change this once we have the actual video file
frame_rate = 120 # Extract 1 frame per second (assuming the video is 30 fps, adjust if needed) basically means we will extract every 30th frame

cap = cv2.VideoCapture(video_path)
frame_embeddings = [] # Memory Storage: List of normalized tensors
frame_paths = [] # To retrieve/display the frames later
frame_count = 200 #short video has beginning frames as useless thats why but otherwise it should start from 0
temp_dir = "temp_frames" # Create a temporary directory to store extracted frames
os.makedirs(temp_dir, exist_ok=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % frame_rate == 0:
        # Save the frame as an image file
        frame_path = os.path.join(temp_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)

        # Preprocess and encode the frame
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = preprocess(pil_image).unsqueeze(0)
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True) # Normalize the features
            frame_embeddings.append(image_features.squeeze(0)) # Store the normalized tensor in memory
    frame_count += 1
cap.release()
print(f"Extracted and encoded {len(frame_embeddings)} frames from the video.")

torch.save({'embeddings': frame_embeddings, 'frame_paths': frame_paths}, embeddings_file)
print(f"Saved embeddings to {embeddings_file} for future retrieval.")
