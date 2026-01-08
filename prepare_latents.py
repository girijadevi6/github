import os
import glob
import torch
from torchvision import transforms
import cv2
import numpy as np
from vae_model import VAE64  # your VAE class file

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "drawing_data"      # folder containing your episodes
LATENT_DIR = "latents"         # folder to save latents
LATENT_DIM = 128

os.makedirs(LATENT_DIR, exist_ok=True)

# Load trained VAE
vae = VAE64(latent_dim=LATENT_DIM).to(DEVICE)
vae.load_state_dict(torch.load("vae_vla_base.pth", map_location=DEVICE))
vae.eval()

# Transform for images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Loop through all images and extract latents
img_paths = glob.glob(os.path.join(DATA_DIR, "*", "episode_*", "*.png"))
print(f"Total images found: {len(img_paths)}")

for i, img_path in enumerate(img_paths):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # 1x1x64x64
    with torch.no_grad():
        _, mu, _ = vae(img_tensor)  # mu is the latent representation
    latent = mu.cpu().numpy()
    # Save latent
    base_name = os.path.basename(img_path).split(".")[0]
    np.save(os.path.join(LATENT_DIR, f"{base_name}.npy"), latent)

print(f"Latents saved in {LATENT_DIR}")
