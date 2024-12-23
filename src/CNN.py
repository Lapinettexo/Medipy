import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data(data_dir, img_size=(128, 128), batch_size=32):
    
    # Define transformations: Resize, normalize, convert to tensor
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizes images to [-1, 1]
    ])

    # Load dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Split into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * (128 // 4) * (128 // 4), 128)  
        self.fc2 = nn.Linear(128, 1)  
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  
        return x



def train_and_evaluate(model, train_loader, test_loader, num_epochs=10, lr=0.001, device='cpu'):
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)  # Labels to float for BCE
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
    
    # Evaluation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = (outputs > 0.5).int()
            correct += (predictions == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)
    
    print(f"Accuracy: {correct / total:.2f}")



data_dir = "C://Users//Trust_pc_dz//Documents//IMED//DATASET//Clean Data"  
img_size = (128, 128)
batch_size = 32


train_loader, test_loader = load_data(data_dir, img_size, batch_size)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BrainTumorCNN()


train_and_evaluate(model, train_loader, test_loader, num_epochs=10, lr=0.001, device=device)


