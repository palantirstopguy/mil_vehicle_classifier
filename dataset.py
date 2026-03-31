import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CSVDataset(Dataset):
    def __init__(self, csv_file, img_dir, filename_col="filename", label_col = "class", transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.filename_col = filename_col
        self.transform = transform

        #clean and filter data
        self._clean_filenames()
        self._filter_existing_files()
        self._setup_labels(label_col)

    def _clean_filenames(self):
        self.df[self.filename_col] = self.df[self.filename_col].astype(str).str.strip()

    def _filter_existing_files(self):
        file_exists = self.df[self.filename_col].apply(lambda f: os.path.exists(os.path.join(self.img_dir, f)))

        missing_count = (~file_exists).sum()
        if missing_count > 0:
            missing_samples = self.df.loc[~file_exists, self.filename_col].head(3).tolist() 
            print(f"skipping{missing_count}missing files")

        self.df = self.df[file_exists].reset_index(drop=True)

    def _setup_labels(self,label_col):
        if self.df[label_col].dtype == object:
            classes = sorted(self.df[label_col.unique()])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            self.idx_to_class = 