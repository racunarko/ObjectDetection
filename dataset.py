import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets


class MNISTLocalization(Dataset):
    def __init__(self, image_size=128, train=False, transform=None):
        self.image_size = image_size
        self.transform = transform
        self.mnist = datasets.MNIST('./data', train=train, download=True)
        self.position_cache = [-1] * len(self.mnist)
     
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        if self.position_cache[idx] == -1:
            x_pos = int(np.random.uniform(0, self.image_size-29))
            y_pos = int(np.random.uniform(0, self.image_size-29))
            self.position_cache[idx] = (x_pos, y_pos)
            
        x_pos, y_pos = self.position_cache[idx]    
        
        image = np.zeros((self.image_size, self.image_size, 1), dtype=np.uint8)
        digit_image = self.mnist[idx][0]
        
        image[y_pos:(y_pos + 28), x_pos:(x_pos + 28), 0] = digit_image

        x_pos = float(x_pos)
        y_pos = float(y_pos)
        
        if self.transform:
            image = self.transform(image)
            
        label = self.mnist[idx][1]
        bbox = [x_pos, y_pos, x_pos+28, y_pos+28]
        
        sample = {'image': image, 'label': label, 'bbox': bbox}
        
        return sample
    