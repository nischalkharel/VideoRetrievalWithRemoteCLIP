#This code is supposed to:
# 1. Load the video from .\data\video_name.mp4
# 2. Extract frames from the video at a specified frame rate
# 3. Preprocess the frames and encode them using the RemoteCLIP model to get image features
# 4. Tokenize the frames and store them in memory
import torch
import open_clip
from PIL import Image
import cv2
import numpy as np
import os
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}\n\n")

#Loading the model first
model_name = "ViT-L-14"
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)
checkpoint_path = "models/RemoteCLIP-ViT-L-14.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)
message = model.load_state_dict(checkpoint)
print(message)
model.to(device)
model.eval()
print("model loaded successfully and ready for inference on: ",device)

embeddings_file = "data/short_video_embeddings.pt"

# Step 1-4: Load the video, extract frames, preprocess, encode them, store embeddings in memory
video_path = "data/short_video.mp4" # Change this once we have the actual video file
frame_interval = 30 # Extract 1 frame per second (assuming the video is 30 fps, adjust if needed) basically means we will extract every 30th frame

cap = cv2.VideoCapture(video_path)
frame_embeddings = [] # Memory Storage: List of normalized tensors
frame_indices = [] # To retrieve/display the frames later
frame_count = 0 #short video has beginning frames as useless thats why but otherwise it should start from 0
start_time = time.time()

frame_batch = []
batch_size = 16 # Adjust based on GPU memory

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % frame_interval == 0:
        # Preprocess and encode the frame
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        preprocessed = preprocess(pil_image).unsqueeze(0)
        frame_batch.append(preprocessed)
        frame_indices.append(frame_count)
        
        if len(frame_batch) == batch_size:
             batch_tensor = torch.cat(frame_batch).to(device)
             with torch.no_grad():
                image_features = model.encode_image(batch_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True) # Normalize the features
                frame_embeddings.extend(image_features.cpu().squeeze(0).tolist()) # Store the normalized tensors in memory
             frame_batch = [] # Clear the batch for the next set of frames
       
    frame_count += 1

if frame_batch: # Process any remaining frames in the batch
    batch_tensor = torch.cat(frame_batch).to(device)
    with torch.no_grad():
        image_features = model.encode_image(batch_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True) # Normalize the features
        frame_embeddings.extend(image_features.cpu().squeeze(0).tolist()) # Store the normalized tensors in memory
        
cap.release()
frame_embeddings_tensor = torch.tensor(frame_embeddings) #converting list of tensors to a single tensor for saving

execution_time = time.time() - start_time
print(f"Extracted and encoded {len(frame_embeddings)} frames from the video.")
print(f"Serial execution time: {execution_time:.2f} seconds")

torch.save({'embeddings': frame_embeddings_tensor, 'frame_paths': frame_indices}, embeddings_file)
print(f"Saved embeddings to {embeddings_file} for future retrieval.")

fps = len(frame_embeddings) / execution_time
print(f"Frames per second (FPS): {fps:.2f}")
