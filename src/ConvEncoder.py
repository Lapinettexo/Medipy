import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


class CustomImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for label in os.listdir(folder_path):
            label_path = os.path.join(folder_path, label)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = plt.imread(img_path)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


folder_path = "C://Users//Trust_pc_dz//Documents//IMED//DATASET//Clean Data"
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])
dataset = CustomImageDataset(folder_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32x32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x16x16
        )
        self.flatten = nn.Flatten()
        self.latent = nn.Linear(64 * 16 * 16, 128)  # Latent features

        # Decoder
        self.decoder_input = nn.Linear(128, 64 * 16 * 16)
        self.unflatten = nn.Unflatten(1, (64, 16, 16))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()  # Final image normalization
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        latent = self.latent(x)
        x = self.decoder_input(latent)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x, latent


model = ConvAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_autoencoder(model, dataloader, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, _ in dataloader:
            images = images.float()
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

train_autoencoder(model, dataloader)


def extract_latent_features(model, dataloader):
    model.eval()
    latent_features = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.float()
            _, latent = model(images)
            latent_features.append(latent)
    return torch.cat(latent_features).numpy()

latent_features = extract_latent_features(model, dataloader)
print("Latent features shape:", latent_features.shape)


kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(latent_features)


cluster_labels = kmeans.labels_
silhouette = silhouette_score(latent_features, cluster_labels)
print("Silhouette Score for clustering:", silhouette)

