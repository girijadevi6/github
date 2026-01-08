import os
import json
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device =", device)

LATENT_FILE = "latents/latents.npy"
ACTION_FILE = "latents/actions.npy"
DATASET_ROOT = "shape_step_dataset"


# ---------------- STEP 1: LOAD LATENTS ----------------
latents = np.load(LATENT_FILE, allow_pickle=True)
print("Latents shape =", latents.shape)


# ---------------- STEP 2: LOAD ACTIONS (SAME LOGIC AS YOUR GRU MODEL) ----------------
raw_actions = np.load(ACTION_FILE, allow_pickle=True)

actions = np.array(
    [a.item()['line'] for a in raw_actions],
    dtype=np.float32
)

print("Actions shape before alignment =", actions.shape)


# ---------- Align Length ----------
if len(actions) == len(latents):
    print(f"⚠️ Actions length == latents length. Truncating actions by 1")
    actions = actions[:len(latents) - 1]

assert len(actions) == len(latents) - 1

# Drop last latent to match actions
latents = latents[:-1]

print("Latents after alignment =", latents.shape)
print("Actions after alignment =", actions.shape)


# ---------------- STEP 3: LOAD LANGUAGE ----------------
instructions = []

for shape in ["square", "rectangle"]:
    shape_dir = os.path.join(DATASET_ROOT, shape)

    for ep in os.listdir(shape_dir):
        ep_dir = os.path.join(shape_dir, ep)
        lang_file = os.path.join(ep_dir, "language.json")

        if not os.path.exists(lang_file):
            continue

        with open(lang_file, "r") as f:
            data = json.load(f)

        for s in data:
            instructions.append(s["instruction"])

# Align instructions too
instructions = instructions[:len(latents)]
print("Loaded instructions =", len(instructions))


# ---------------- STEP 4: TEXT EMBEDDINGS ----------------
print("Encoding language...")
text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
text_embeds = text_encoder.encode(instructions)

text_embeds = torch.tensor(text_embeds, dtype=torch.float32).to(device)


# ---------------- Convert to Torch ----------------
latents = torch.tensor(latents, dtype=torch.float32).to(device)
actions = torch.tensor(actions, dtype=torch.float32).to(device)

latent_dim = latents.shape[1]
text_dim = text_embeds.shape[1]
action_dim = actions.shape[1]

print("latent_dim =", latent_dim)
print("text_dim =", text_dim)
print("action_dim =", action_dim)


# ---------------- STEP 5: POLICY MODEL ----------------
class VLA_Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + text_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, z, t):
        x = torch.cat([z, t], dim=-1)
        return self.net(x)


policy = VLA_Policy().to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()


# ---------------- STEP 6: DATASET ----------------
dataset = torch.utils.data.TensorDataset(latents, text_embeds, actions)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


# ---------------- STEP 7: TRAIN ----------------
EPOCHS = 50

for epoch in range(EPOCHS):
    total_loss = 0

    for z, t, a in loader:
        pred = policy(z, t)
        loss = loss_fn(pred, a)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}  Loss = {total_loss/len(loader):.6f}")


# ---------------- SAVE ----------------
torch.save(policy.state_dict(), "vla_policy.pt")
print("\n🎉 Training Complete — Saved as vla_policy.pt")
