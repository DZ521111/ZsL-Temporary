import os
import pickle
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Define the path to your dataset and output file
dataset_path = '/Users/dhruvkumarkakadiya/Desktop/IITG Data/DL DUET/dataset/CUB_200_2011/CUB_200_2011/images'
output_pkl_file = './id2imagepixel.pkl'

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the dataset using torchvision
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Initialize a dictionary to store image features
id2imagepixel = {}

# Use a pre-trained model to extract features (if needed)
# For this example, we're using raw image tensors
for idx, (image, _) in enumerate(dataloader):
    image_id = dataset.imgs[idx][0].split('/')[-1]  # Use image file name as ID
    image_tensor = image.numpy()
    id2imagepixel[image_id] = image_tensor

# Save the dictionary to a .pkl file
with open(output_pkl_file, 'wb') as f:
    pickle.dump(id2imagepixel, f)

print(f"Generated .pkl file and saved to {output_pkl_file}")
