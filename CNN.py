import os
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CustomDataset(Dataset):
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')

        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Define your CNN architecture
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(8 * 350 * 350, 64)  # (8 channels * height * width)
        self.fc2 = nn.Linear(64, 1)  # Single neuron for prediction

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 8 * 350 * 350)  # Flatten the tensor based on the new size
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    data_directory = "./data"
    
    transform = transforms.Compose([
        transforms.Resize((700, 700)),  # Resize the images to a consistent size
        transforms.ToTensor(),
    ])
    
    # Load image paths and labels from CSV
    data_df = pd.read_csv(os.path.join(data_directory, "image_data.csv"))
    image_paths = [os.path.join(data_directory, "output", img) for img in data_df.iloc[:, 0]]
    labels = data_df.iloc[:, 1].astype(int).tolist()
    
    # Manually split data into training and testing sets
    split_ratio = 0.1  # 80% for training, 20% for testing
    split_index = int(len(image_paths) * split_ratio)
    train_image_paths, test_image_paths = image_paths[:split_index], image_paths[split_index:]
    train_labels, test_labels = labels[:split_index], labels[split_index:]
    
    # Create datasets for training and testing
    train_dataset = CustomDataset(image_paths=train_image_paths, labels=train_labels, transform=transform)
    test_dataset = CustomDataset(image_paths=test_image_paths, labels=test_labels, transform=transform)
    
    batch_size = 5
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    
    
    # Instantiate the model and move it to the device
    model = CNNModel().to(device)
    
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        for batch_images, batch_labels in train_dataloader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.float().to(device)  # Convert labels to float
            
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels.view(-1, 1))  # Reshape labels for MSE loss
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            print(f"LOSS: {running_loss}")
        
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_dataloader)}")
    
    print("Training Finished!")
    
    model_path = "trained_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved at {model_path}")
    """
    
    # Load the trained model for testing
    loaded_model = CNNModel().to(device)
    loaded_model.load_state_dict(torch.load("trained_model.pth"))
    loaded_model.eval()  # Set the model to evaluation mode

    threshold = 3  # Adjust this threshold based on your use case
    
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch_images, batch_labels in test_dataloader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.float().to(device)

            outputs = loaded_model(batch_images)

            # Calculate absolute differences between predictions and labels
            differences = torch.abs(outputs - batch_labels.view(-1, 1))

            # Count correct predictions based on the threshold
            correct_predictions += torch.sum(differences <= threshold).item()
            total_samples += batch_labels.size(0)

    accuracy = correct_predictions / total_samples * 100.0

    print(f"Accuracy: {accuracy:.2f}%")
    """
        
if __name__ == '__main__':
    main()