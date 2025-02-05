# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:51:54 2025

@author: aron-
"""

import os
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt

# Set folder path
folder_path = r"C:\Users\aron-\Desktop\Thesis\data\AronDataset\Field31.03.2022\101904_Posmico_campo_2_2015_06_04_18_30_35"

# Find all .hdr files (metadata for hyperspectral images)
hdr_files = [f for f in os.listdir(folder_path) if f.endswith(".hdr")]

# Dictionary to store reflectance spectra
reflectance_data = {}

# Load each hyperspectral dataset
for hdr_file in hdr_files:
    # Get corresponding raw file (assuming it's the same name but without .hdr)
    raw_file = hdr_file.replace(".hdr", "")
    raw_path = os.path.join(folder_path, raw_file)
    hdr_path = os.path.join(folder_path, hdr_file)

    # Load the hyperspectral image
    try:
        img = envi.open(hdr_path, raw_path).load()
    except Exception as e:
        print(f"Skipping {hdr_file}: {e}")
        continue

    # Select a pixel (for example, center of the image)
    center_pixel = img.shape[0] // 2, img.shape[1] // 2
    spectrum = img[center_pixel[0], center_pixel[1], :]  # Extract spectrum at center pixel

    # Store reflectance data
    reflectance_data[hdr_file] = spectrum

# Plot reflectance spectra
wavelengthes= np.load(r"C:\Users\aron-\Desktop\Thesis\thesis_git\programming\wavelengthes.npy")
spectrum = reflectance_data['raw_0.hdr']
spectrum = np.squeeze(spectrum)
plt.figure(figsize=(10, 5))
plt.plot(wavelengthes, spectrum)
# for filename, spectrum in reflectance_data.items():
#     plt.plot(spectrum, label=filename)

plt.xlabel("Wavelength (Bands)")
plt.ylabel("Reflectance")
plt.title("Reflectance Spectra from Hyperspectral Images")
plt.legend()
plt.grid()
plt.show()


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Select multispectral bands from hyperspectral data (example indices)
MULTISPECTRAL_BANDS = [10, 50, 100, 150, 200, 250]

class HyperspectralDataset(Dataset):
    def __init__(self, hyperspectral_data, multispectral_bands):
        self.data = []
        for img in hyperspectral_data.values():
            for key in img.keys():
                hs_img = img[key]  # Shape: (H, W, C)
                ms_img = hs_img[:, :, multispectral_bands]  # Extract multispectral bands
                self.data.append((ms_img, hs_img))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ms, hs = self.data[idx]
        return torch.tensor(ms, dtype=torch.float32), torch.tensor(hs, dtype=torch.float32)

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, output_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
batch_size = 8
epochs = 50
learning_rate = 0.0002

# Load dataset
dataset = HyperspectralDataset(hyperspectral_data, MULTISPECTRAL_BANDS)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
generator = Generator(len(MULTISPECTRAL_BANDS), 272)  # 272 hyperspectral bands
discriminator = Discriminator(272)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

def train():
    for epoch in range(epochs):
        for ms, hs in dataloader:
            ms, hs = ms.permute(0, 3, 1, 2), hs.permute(0, 3, 1, 2)  # Reshape for CNN

            # Train Discriminator
            real_labels = torch.ones(ms.size(0), 1, 1, 1)
            fake_labels = torch.zeros(ms.size(0), 1, 1, 1)

            optimizer_d.zero_grad()
            real_outputs = discriminator(hs)
            real_loss = criterion(real_outputs, real_labels)

            fake_hs = generator(ms)
            fake_outputs = discriminator(fake_hs.detach())
            fake_loss = criterion(fake_outputs, fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            fake_outputs = discriminator(fake_hs)
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

train()
