from spectral import imshow, view_cube
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

load_all_images = 'no'
rgb_normalize = 'no'

# Define the main directories

base_dirs = [
    r"C:\Users\aron-\Desktop\Thesis\data\AronDataset\Field31.03.2022",
    r"C:\Users\aron-\Desktop\Thesis\data\AronDataset\Field06.04.2022"
]


# Dictionary to store loaded hyperspectral images
hyperspectral_data = {}

for base_path in base_dirs:
    # Get all subfolders in the base directory
    subfolders = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    for folder in subfolders:
        # Get all hyperspectral image files (assuming ENVI format, .hdr files)
        hdr_files = [f for f in os.listdir(folder) if f.endswith(".hdr")]
        key = f"{os.path.basename(base_path)}_{os.path.basename(folder)}"
        hyperspectral_data[key] = {}
        for hdr in hdr_files:
            img_path = os.path.join(folder, hdr)
            
            raw_file = hdr.replace(".hdr", "")
            raw_path = os.path.join(folder, raw_file)
            hdr_path = os.path.join(folder, hdr)

            # Load hyperspectral image
            try:
                img = envi.open(hdr_path, raw_path).load()
            except Exception as e:
                print(f"Skipping {hdr}: {e}")
                continue
            
            
            #RGB image
            if rgb_normalize == 'yes':
                rgb_path = hdr.replace(".hdr", ".png")
                rgb_path = os.path.join(folder, rgb_path)
                rgb = plt.imread(rgb_path) 

                # Normalize hyperspectral image using max RGB intensity per pixel
                rgb_gray = np.mean(rgb, axis=2, keepdims=True)  # Convert RGB to grayscale intensity
                rgb_gray[rgb_gray == 0] = 1  

                # Normalize each hyperspectral band
                reflectance = img / rgb_gray 

                # Clip values to [0,1]
                img = np.clip(reflectance, 0, 1)


            # Create a unique key based on field date, subfolder, and filename
            # key = f"{os.path.basename(base_path)}_{os.path.basename(folder)}_{hdr}"
            hyperspectral_data[key][hdr] = img

        if load_all_images == 'no':
            break
# Print loaded dataset keys
print("Loaded images:", hyperspectral_data.keys())
print("Loaded images:", hyperspectral_data['Field31.03.2022_101904_Posmico_campo_2_2015_06_04_18_30_35'].keys())
#hyperspectral_data['Field31.03.2022_101904_Posmico_campo_2_2015_06_04_18_30_35']['raw_0.hdr']

wavelengthes= np.load(r"C:\Users\aron-\Desktop\Thesis\thesis_git\programming\wavelengthes.npy")

#shape (2000, 640, 272)
spectrum = hyperspectral_data['Field31.03.2022_101904_Posmico_campo_2_2015_06_04_18_30_35']['raw_0.hdr'][100, 100, :]

spectrum = np.squeeze(spectrum)
plt.figure(figsize=(10, 5))
plt.plot(wavelengthes, spectrum)
plt.xlabel("Wavelength (Bands)")
plt.ylabel("Reflectance")
plt.title("Reflectance Spectra from Hyperspectral Images")
plt.legend()
plt.grid()
plt.show()


#small model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# Select multispectral bands from hyperspectral data (example indices)
MULTISPECTRAL_BANDS = [10, 34, 50, 62, 100, 131, 150, 191, 200, 250]

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
    mse_losses = []
    for epoch in range(epochs):
        epoch_mse_loss = 0
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

            # Compute MSE loss
            mse_loss = torch.nn.functional.mse_loss(fake_hs, hs)
            epoch_mse_loss += mse_loss.item()
        
        mse_losses.append(epoch_mse_loss / len(dataloader))
        print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | MSE Loss: {mse_losses[-1]:.4f}")

    # Plot MSE loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), mse_losses, label="MSE Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Mean Squared Error Loss During Training")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot example comparison
    ms, hs = next(iter(dataloader))
    ms, hs = ms.permute(0, 3, 1, 2), hs.permute(0, 3, 1, 2)
    fake_hs = generator(ms).detach().cpu().numpy()
    hs = hs.cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(hs[0, :, :, 0].flatten(), label="True Hyperspectral")
    plt.plot(fake_hs[0, :, :, 0].flatten(), label="Predicted Hyperspectral")
    plt.xlabel("Pixel Index")
    plt.ylabel("Reflectance")
    plt.title("True vs Predicted Hyperspectral Data")
    plt.legend()
    plt.grid()
    plt.show()

train()


