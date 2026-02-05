import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import re

class FilenameRegressDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        # List all .jpg files in the directory
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        # Returns the total number of samples
        return len(self.file_list)

    def __getitem__(self, idx):
        # 1. Get the filename
        img_name = self.file_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # 2. Open the image
        image = Image.open(img_path).convert('RGB')
        
        # 3. Parse the filename to get the label
        # Example: "000229-0.50.jpg"

        # The Pattern:
        # ^          -> Start of the string
        # (.*_data_) -> Group 1: Matches anything ending in "_data_"
        # (.*)       -> Group 2: Matches everything else (the original name)
        # $          -> End of the string
        pattern = r"^(.*_data_)(.*)$"

        match = re.match(pattern, img_name)
        if match:
            img_name = match.group(2)

        # Remove the extension (removes ".jpg") -> "000229-0.50"
        name_no_ext = os.path.splitext(img_name)[0]
        
        # Extract the value part. 
        # You said the first 6 digits are the ID.
        # So we slice from index 6 to the end.
        label_str = name_no_ext[6:] 
        
        # Convert string "-0.50" to float -0.5
        label = float(label_str) 
        
        # 4. Convert label to a Tensor (shape needs to be [1] for regression)
        label_tensor = torch.tensor([label], dtype=torch.float32)

        # 5. Apply image transforms (resize, normalize, etc.)
        if self.transform:
            image = self.transform(image)

        return image, label_tensor

if __name__ == "__main__":
    # --- How to use it ---

    # Define transformations (Resizing is crucial so all images are same shape)
    data_transform = transforms.Compose([
        # Crop the image to focus on relevant area (50 % from top)
        transforms.Lambda(lambda x: x.crop((0, x.size[1] // 2, x.size[0], x.size[1]))),
        transforms.Resize((120, 320)), # Resize to fit your model input
        transforms.ToTensor(),
    ])

    # Instantiate the dataset
    # Replace './my_images' with the actual path to your folder
    dataset = FilenameRegressDataset(root_dir='./data/train', transform=data_transform)

    # Create the DataLoader
    # batch_size=32 means it gives you 32 images at a time
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Fetch one batch
    images, labels = next(iter(train_loader))

    # Show image
    import matplotlib.pyplot as plt
    plt.imshow(images[0].permute(1, 2, 0))
    plt.title(f"Label: {labels[0].item()}")
    plt.show()

    print(f"Batch shape: {images.shape}") # Should be [32, 3, 64, 64]
    print(f"Labels shape: {labels.shape}") # Should be [32, 1]

    # Print first 5 parsed labels to check against your filenames
    print("\nFirst 5 labels in this batch:")
    print(labels[:5])