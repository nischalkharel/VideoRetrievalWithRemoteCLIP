import torch
import open_clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "ViT-L-14"
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
checkpoint = torch.load("models/RemoteCLIP-ViT-L-14.pt", map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# Disable fused attention fast path
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

dummy_input = torch.randn(1, 3, 224, 224).to(device)

with torch.no_grad():
    scripted = torch.jit.trace(model.visual, dummy_input)
    scripted.save("models/RemoteCLIP-ViT-L-14_scripted.pt")

print("Done.")