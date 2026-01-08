import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# -----------------------
# Dataset
# -----------------------
class WorldModelDataset(Dataset):
    def __init__(self, latent_file="latents/latents.npy", action_file="latents/actions.npy"):
        self.latents = np.load(latent_file).astype(np.float32)

        raw_actions = np.load(action_file, allow_pickle=True)
        self.actions = np.array([a.item()['line'] for a in raw_actions], dtype=np.float32)

        if len(self.actions) == len(self.latents):
            print(
                f"Warning: actions length ({len(self.actions)}) == latents length ({len(self.latents)}). "
                f"Truncating actions by 1."
            )
            self.actions = self.actions[:len(self.latents) - 1]

        assert len(self.actions) == len(self.latents) - 1

        self.action_dim = self.actions.shape[1]

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        z_t = torch.tensor(self.latents[idx], dtype=torch.float32)
        a_t = torch.tensor(self.actions[idx], dtype=torch.float32)
        z_next = torch.tensor(self.latents[idx + 1], dtype=torch.float32)
        return z_t, a_t, z_next


# -----------------------
# GRU World Model (1-step)
# -----------------------
class GRUWorldModel(nn.Module):
    def __init__(self, latent_dim=128, action_dim=4, hidden_dim=256):
        super().__init__()

        self.input_dim = latent_dim + action_dim

        self.gru = nn.GRUCell(
            input_size=self.input_dim,
            hidden_size=hidden_dim
        )

        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z, a, h=None):
        """
        z: (B, latent_dim)
        a: (B, action_dim)
        """
        x = torch.cat([z, a], dim=1)  # (B, latent+action)

        if h is None:
            h = torch.zeros(z.size(0), self.gru.hidden_size, device=z.device)

        h_next = self.gru(x, h)       # (B, hidden_dim)
        z_pred = self.fc(h_next)      # (B, latent_dim)

        return z_pred, h_next


# -----------------------
# Trainer
# -----------------------
class WorldModelTrainer:
    def __init__(self, model, device, lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train(self, loader, epochs=50):
        self.model.train()

        for epoch in range(1, epochs + 1):
            total_loss = 0

            for z_t, a_t, z_next in tqdm(loader, desc=f"Epoch {epoch}"):
                z_t = z_t.to(self.device)
                a_t = a_t.to(self.device)
                z_next = z_next.to(self.device)

                self.optimizer.zero_grad()

                z_pred, _ = self.model(z_t, a_t)
                loss = self.criterion(z_pred, z_next)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch} - Avg Loss: {total_loss / len(loader):.6f}")

    def save(self, path="world_model_gru.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved at {path}")


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    LATENT_FILE = "latents/latents.npy"
    ACTION_FILE = "latents/actions.npy"
    LATENT_DIM = 128
    HIDDEN_DIM = 256
    BATCH_SIZE = 32
    EPOCHS = 25

    dataset = WorldModelDataset(LATENT_FILE, ACTION_FILE)
    ACTION_DIM = dataset.action_dim

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRUWorldModel(
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM
    )

    trainer = WorldModelTrainer(model, device)
    trainer.train(loader, epochs=EPOCHS)
    trainer.save("world_model_gru_25.pth")
