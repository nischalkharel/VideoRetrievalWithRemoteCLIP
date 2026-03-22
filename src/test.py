import open_clip

model_name = "ViT-L-14"
model, _, preprocess = open_clip.create_model_and_transforms(model_name)

print(preprocess)