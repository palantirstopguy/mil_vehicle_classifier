import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CSVDataset(Dataset):
    def __init__(self, csv_file, img_dir, filename_col="filename", label_col = "class", transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.filename_col