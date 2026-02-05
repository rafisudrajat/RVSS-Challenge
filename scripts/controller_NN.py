import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import FilenameRegressDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime


# 1. Define the Neural Network
class ImageRegressor(nn.Module):
    def __init__(self):
        super(ImageRegressor, self).__init__()
        
        # Feature Extractor (Convolutional Layers)
        # Input assumes a 3-channel image (RGB)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Image size halves
            
            # Block 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Image size halves again
            
            # Block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Image size halves again
        )
        
        # Regressor (Fully Connected Layers)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            # Calculation for input size:
            nn.Linear(38400, 128), 
            nn.ReLU(),
            # Now this matches the 128 from above
            nn.Linear(128, 1) # Output layer: 1 neuron for a single real number
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

if __name__ == "__main__":
    # Get the current time
    now = datetime.now()

    # Format it as DD_MM_YYYY_HH_MM
    timestamp = now.strftime("%d_%m_%Y_%H_%M")
    
    # --- Main Execution Block ---
    # Setup Device (GPU if available)
    EPOCH_NUM = 30
    BATCH_SIZE = 16
    LR = 0.001

    MODEL_PATH = './model/model_' + timestamp + '.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = ImageRegressor().to(device)

    # Define Loss and Optimizer
    # MSELoss is the standard for regression tasks
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Starting Training Loop...")
    model.train()

    # Define transformations (Resizing is crucial so all images are same shape)
    data_transform = transforms.Compose([
        # Crop the image to focus on relevant area (50 % from top)
        transforms.Lambda(lambda x: x.crop((0, x.size[1] // 2, x.size[0], x.size[1]))),
        transforms.Resize((120, 320)), # Resize to fit your model input
        transforms.ToTensor(),
    ])

    # Instantiate the dataset
    # Replace './my_images' with the actual path to your folder
    train_dataset = FilenameRegressDataset(root_dir='./data/train_combines', transform=data_transform)
    test_dataset = FilenameRegressDataset(root_dir='./data/val_starter', transform=data_transform)

    # Create the DataLoader
    # batch_size=32 means it gives you 32 images at a time
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    for epoch in range(EPOCH_NUM): # Run for EPOCH_NUM epochs
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Move data to device
            images = images.to(device)
            targets = targets.to(device)


            # A. Forward pass: Compute predicted y by passing x to the model
            outputs = model(images)
            
            # B. Compute and print loss
            loss = criterion(outputs, targets)
        
            # C. Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{EPOCH_NUM}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), MODEL_PATH)

    # --- Evaluation on Test Set ---
    print("Evaluating on test set...")
    total_testing_loss = 0.0
    model.eval()
    for batch_idx, (images, targets) in enumerate(test_loader):
        images = images.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(images)
            print("predicted:", outputs.squeeze().cpu().numpy())
            print("target:", targets.squeeze().cpu().numpy())
            loss = criterion(outputs, targets)
            total_testing_loss += loss.item()
        
        print(f'Test Batch [{batch_idx+1}/{len(test_loader)}], Loss: {loss.item():.4f}')
    print(f"Average Test Loss: {total_testing_loss / len(test_loader):.4f}")