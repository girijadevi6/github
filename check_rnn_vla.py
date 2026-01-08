import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from vla_rnn import VLA_RNN_Policy  # import your GRU class

# --------------------------- 1️⃣ Device setup ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device =", device)

# --------------------------- 2️⃣ Load latents ---------------------------
latents = np.load("latents/latents.npy")
num_total_states = len(latents)
num_episodes = 4          # your dataset: 4 episodes
steps_per_episode = num_total_states // num_episodes
print(f"Loaded {num_total_states} latents ({num_episodes} episodes x {steps_per_episode} steps each)")

# --------------------------- 3️⃣ Load text encoder ---------------------------
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------- 4️⃣ Get user input ---------------------------
episode_index = int(input(f"Enter episode index (0 to {num_episodes-1}): "))
step_index = int(input(f"Enter step index (0 to {steps_per_episode-1}): "))
instruction = input("Enter instruction (e.g., 'extend line downward'): ")

# Validate indices
if episode_index < 0 or episode_index >= num_episodes:
    raise ValueError(f"Episode index must be between 0 and {num_episodes-1}")
if step_index < 0 or step_index >= steps_per_episode:
    raise ValueError(f"Step index must be between 0 and {steps_per_episode-1}")

# --------------------------- 5️⃣ Compute flattened latent index ---------------------------
state_index = episode_index * steps_per_episode + step_index
print(f"Using latent index {state_index} (episode {episode_index}, step {step_index})")

# --------------------------- 6️⃣ Prepare latent tensor ---------------------------
z = torch.tensor(latents[state_index], dtype=torch.float32).unsqueeze(0).to(device)  # [1, latent_dim]

# --------------------------- 7️⃣ Encode instruction ---------------------------
text = encoder.encode(instruction, convert_to_tensor=True).unsqueeze(0).to(device)  # [1, text_dim]

# --------------------------- 8️⃣ Load RNN policy ---------------------------
latent_dim = z.shape[1]
text_dim = text.shape[1]
action_dim = 4  # replace with your action dimension

policy = VLA_RNN_Policy(latent_dim=latent_dim, text_dim=text_dim, action_dim=action_dim).to(device)
policy.load_state_dict(torch.load("vla_rnn_policy.pt", map_location=device))
policy.eval()

# --------------------------- 9️⃣ Prepare sequence for single-step prediction ---------------------------
z_seq = z.unsqueeze(1)    # [1, seq_len=1, latent_dim]
t_seq = text.unsqueeze(1) # [1, seq_len=1, text_dim]

# --------------------------- 🔟 Predict action ---------------------------
with torch.no_grad():
    action_seq, _ = policy(z_seq, t_seq)  # [1, seq_len, action_dim]
    action = action_seq[:, 0, :]          # extract the single step

# --------------------------- 1️⃣1️⃣ Print predicted action ---------------------------
print("Predicted Action:", action.cpu().numpy())
