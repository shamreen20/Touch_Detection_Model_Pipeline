import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
try:
    from pytouch import PyTouch, TouchDetect
except ImportError:
    print("PyTouch not available. Skipping PyTouch baseline.")
    PyTouch = None

# Custom Dataset for Grasping Trials
class GraspingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        # Iterate through subdirectories in collected_data
        for subdir in os.listdir(data_dir):
            subdir_path = os.path.join(data_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue


            # Load labels for this trial
            label_file = os.path.join(subdir_path, 'labels_supervised.npy')
            if not os.path.exists(label_file):
                print(f"Warning: Labels file not found at {label_file}")
                continue
            labels = np.load(label_file)
             # Handle 1D or 2D label arrays
            if labels.ndim == 1:
                label = labels[0]  # Single value for grasp success
            elif labels.ndim == 2:
                label = labels[0, 0]  # First column for grasp success
                print(f"Warning: Empty labels in {label_file}")
                continue

            # Collect image paths for middle and thumb
            for finger in ['middle', 'thumb']:
                img_path = os.path.join(subdir_path, f'touch_{finger}_3.png')
                if os.path.exists(img_path):
                    self.image_files.append(img_path)
                    self.labels.append(labels[0])  # Single label per trial
                else:
                    print(f"Warning: Image not found at {img_path}")

        if len(self.image_files) == 0:
            raise ValueError("No valid images found in the dataset")
        print(f"Loaded {len(self.image_files)} image-label pairs")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels[idx]

        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

# CNN Model
class TouchCNN(nn.Module):
    def __init__(self):
        super(TouchCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),  # Assumes 128x128 input
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# MLP Model
class TouchMLP(nn.Module):
    def __init__(self, input_size=128*128*3):
        super(TouchMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

# Training Function
def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%')

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).squeeze()
                predicted = (outputs >= 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        print(f'Validation Accuracy: {val_accuracy:.2f}%')

# Main Pipeline
def main():
    # Data Preparation
    data_dir = './collected_data/'  
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    try:
        dataset = GraspingDataset(data_dir, transform=transform)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Train CNN Model
    print("Training CNN Model...")
    cnn_model = TouchCNN()
    train_model(cnn_model, train_loader, val_loader, num_epochs=10, device=device)

    # Train MLP Model
    print("\nTraining MLP Model...")
    mlp_model = TouchMLP()
    train_model(mlp_model, train_loader, val_loader, num_epochs=10, device=device)

    # PyTouch Baseline (if available)
    if PyTouch:
        print("\nRunning PyTouch Baseline...")
        try:
            pytouch_model = PyTouch(tasks=[TouchDetect])
            # Note: PyTouch may not support the dataset format directly
            for images, _ in train_loader:
                images_np = images.permute(0, 2, 3, 1).numpy()
                for img in images_np:
                    pytouch_model.process(img)  # Placeholder processing
            print("PyTouch baseline completed (custom integration may be needed).")
        except Exception as e:
            print(f"PyTouch baseline failed: {e}")
    else:
        print("PyTouch not installed. Skipping baseline.")

if __name__ == '__main__':
    main()