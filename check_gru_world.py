import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from world_model_gru import GRUWorldModel   # <-- your GRU world model file
from vae_model import VAE64                 # <-- your VAE

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load Latents + Actions
# -----------------------
latents = np.load("latents/latents.npy").astype(np.float32)

raw_actions = np.load("latents/actions.npy", allow_pickle=True)
actions = np.array([a.item()['line'] for a in raw_actions], dtype=np.float32)

# -----------------------
# Load VAE
# -----------------------
vae = VAE64().to(device)
vae.load_state_dict(torch.load("vae_vla_base.pth", map_location=device))
vae.eval()

# -----------------------
# Load GRU World Model
# -----------------------
LATENT_DIM = latents.shape[1]
ACTION_DIM = actions.shape[1]
HIDDEN_DIM = 256

world = GRUWorldModel(
    latent_dim=LATENT_DIM,
    action_dim=ACTION_DIM,
    hidden_dim=HIDDEN_DIM
).to(device)

world.load_state_dict(torch.load("world_model_gru_25.pth", map_location=device))
world.eval()

# -----------------------
# Select Example
# -----------------------
z_index = 9        # current latent
a_index = 10        # action applied to z_index

z_t = torch.tensor(latents[z_index]).unsqueeze(0).to(device)
a_t = torch.tensor(actions[a_index]).unsqueeze(0).to(device)

# -----------------------
# Predict Next Latent
# -----------------------
with torch.no_grad():
    z_next_pred, _ = world(z_t, a_t)

# -----------------------
# Decode Images
# -----------------------
with torch.no_grad():
    img_current = vae.decode(z_t)
    img_pred = vae.decode(z_next_pred)

# -----------------------
# Convert for plotting
# -----------------------
img_current_np = img_current.squeeze(0).cpu().permute(1, 2, 0).numpy()
img_pred_np = img_pred.squeeze(0).cpu().permute(1, 2, 0).numpy()

# -----------------------
# Visualization
# -----------------------
plt.figure(figsize=(10, 4))

# Current State
plt.subplot(1, 3, 1)
plt.imshow(img_current_np, cmap="gray")
plt.title("Current State (z_t)")
plt.axis("off")

# Action
plt.subplot(1, 3, 2)
plt.axis("off")
action_text = "\n".join([f"{v:.3f}" for v in a_t.squeeze(0).cpu().numpy()])
plt.text(0.1, 0.5, f"Action (line):\n{action_text}", fontsize=12)
plt.title("Action a_t")

# Predicted State
plt.subplot(1, 3, 3)
plt.imshow(img_pred_np, cmap="gray")
plt.title("Predicted Next State (ẑₜ₊₁)")
plt.axis("off")

plt.tight_layout()
plt.savefig("world_model_gru_25.png")
plt.show()

print("Saved visualization as world_model_gru_demo.png")

# -----------------------
# Optional: save images separately
# -----------------------
vutils.save_image(img_current, "current_state.png")
vutils.save_image(img_pred, "predicted_next_state.png")
