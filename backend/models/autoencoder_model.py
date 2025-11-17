import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(3, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Conv2d(64, 128, 3, padding=1)

        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Conv2d(64, 64, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Conv2d(32, 32, 3, padding=1)
        self.up3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.out = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = self.pool(F.relu(self.enc2(x)))  # 16x16
        x = self.pool(F.relu(self.enc3(x)))  # 8x8

        x = F.relu(self.up1(x))             # 16x16
        x = F.relu(self.dec1(x))
        x = F.relu(self.up2(x))             # 32x32
        x = F.relu(self.dec2(x))
        x = F.relu(self.up3(x))             # 64x64 -> but input is 32x32; to keep size, adjust
        # To keep CIFAR-10 at 32x32, we avoid last upsample; instead clamp shape handling
        return torch.sigmoid(self.out(x))


def build_autoencoder() -> nn.Module:
    # Simpler: keep consistent 32x32 by adjusting architecture
    class _AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Conv2d(3, 32, 3, padding=1)
            self.enc2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.enc3 = nn.Conv2d(64, 128, 3, padding=1)

            self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 8->16
            self.dec1 = nn.Conv2d(64, 64, 3, padding=1)
            self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)   # 16->32
            self.dec2 = nn.Conv2d(32, 32, 3, padding=1)
            self.out = nn.Conv2d(32, 3, 3, padding=1)

        def forward(self, x):
            x = F.relu(self.enc1(x))
            x = self.pool(F.relu(self.enc2(x)))  # 16x16
            x = self.pool(F.relu(self.enc3(x)))  # 8x8
            x = F.relu(self.up1(x))
            x = F.relu(self.dec1(x))
            x = F.relu(self.up2(x))
            x = F.relu(self.dec2(x))
            x = torch.sigmoid(self.out(x))
            return x

    return _AE()


def train_autoencoder(model: nn.Module, dataset: TensorDataset, device: torch.device, epochs: int = 5, batch_size: int = 128, lr: float = 1e-3) -> None:
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True if device.type == "cuda" else False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, _ in loader:
            xb = xb.to(device)
            recon = model(xb)
            loss = F.mse_loss(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()



