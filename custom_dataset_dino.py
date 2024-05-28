import json
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

class CustomSSLDataset(Dataset):
    def __init__(self, json_file_path, transform):
        self.data = self.load_data(json_file_path)
        self.transform = transform

    def load_data(self, json_file_path):
        # Load your JSON file
        # Assuming you're using the 'json' module
        import json
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load and preprocess your data here
        # For example, loading an image using PIL and applying transformations
        img_path_c = self.data[idx]['file_name']
        #img_path = '/home/woody/iwfa/iwfa047h/Winding_Heads/' + img_path
        img_path = '/home/woody/iwfa/iwfa048h/Python-Code/database/data_processed/Classification/Cable/' + img_path_c
        img = Image.open(img_path).convert('RGB')

        # You can apply any other transformations here if needed
        if self.transform:
            img = self.transform(img)

        # Extract other relevant data
        label = int(self.data[idx]['label'])

        return img, label, img_path_c