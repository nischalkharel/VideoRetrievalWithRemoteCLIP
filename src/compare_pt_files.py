import torch
import open_clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load original model
model_name = "ViT-L-14"
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
checkpoint = torch.load("models/RemoteCLIP-ViT-L-14.pt", map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# Load scripted model
scripted = torch.jit.load("models/RemoteCLIP-ViT-L-14_scripted.pt")
scripted.to(device)
scripted.eval()

# Use a fixed seed so both get identical input
torch.manual_seed(42)
dummy_input = torch.randn(1, 3, 224, 224).to(device)

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

with torch.no_grad():
    out_original = model.visual(dummy_input)
    out_scripted = scripted(dummy_input)

# Check 1: Shape
print(f"Original shape: {out_original.shape}")
print(f"Scripted shape: {out_scripted.shape}")

# Check 2: Max absolute difference (should be near 0, tiny float tolerance is fine)
max_diff = (out_original - out_scripted).abs().max().item()
print(f"Max absolute difference: {max_diff}")

# Check 3: Are they close within float32 tolerance?
are_close = torch.allclose(out_original, out_scripted, atol=1e-5, rtol=1e-5)
print(f"Outputs match (within tolerance): {are_close}")

# Check 4: Cosine similarity (most meaningful for embeddings)
cos_sim = torch.nn.functional.cosine_similarity(
    out_original.float(), out_scripted.float()
).item()
print(f"Cosine similarity: {cos_sim:.8f}")  # should be ~1.0000000