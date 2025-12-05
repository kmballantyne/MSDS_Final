import os
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from config import *



class CustomDataset(Dataset):
    def __init__(
        self, 
        csv_file: str,
        label_col: str, 
        client_dir: str | None = None, 
        path_col: str = "local_path", 
        transform=None
        ):
            """
            Args:
                csv_file: path to client CSV (e.g. client1_train_clean.csv)
                label_col: name of label column (e.g. "Pleural Effusion")
                transform: torchvision transforms to apply
                path_col: column in CSV with image paths ("local_path" for you)
                client_dir: optional root directory for relative paths
            """
            self.data = pd.read_csv(csv_file)
            self.client_dir = client_dir
            self.label_col = label_col
            self.path_col = path_col
            self.transform = transform
            
            if self.path_col not in self.data.columns:
                raise ValueError(
                    f"CSV {csv_file} missing required column: {self.path_col}"
                    f"Available columns: {list(self.data.columns)}"
                )
            
            # Log number of positive examples
            pos_count = sum(self.data[self.label_col].astype(str).str.strip() == "1.0")
            print(f"[INFO] Loaded dataset from {csv_file} with {pos_count} positives for label '{self.label_col}'.")
    
    def __len__(self):
        return len(self.data)
    
    def _get_image_path(self, idx):
        """Resolve full image path from CSV row."""
        row = self.data.iloc[idx]
        img_path = str(row[self.path_col]).strip()

        # If path is relative and a root is provided, join them
        if not os.path.isabs(img_path) and self.client_dir is not None:
            img_path = os.path.join(self.client_dir, img_path)

        return img_path

    def __getitem__(self, idx):
        img_path = self._get_image_path(idx)
        # img_rel_path = self.data.iloc[idx]['Path']
        # img_path = os.path.join(self.client_dir, img_rel_path)

        label = self.data.iloc[idx][self.label_col]
        label = str(label).strip()
        if label == "1.0":
            label = 1
        else: # handles 0.0, -1.0, "", NaN
            label = 0

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.data)) # recursively try next item on error

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label).long()

def load_client_datasets(
    base_dir=METADATA_DIR,
    label_col="Pleural_Effusion",
):
    """
    Loads one dataset per client.

    CSV files are assumed to be:
        client1_train_clean.csv
        client2_train_clean.csv
        ...
        client5_train_clean.csv

    Each CSV contains a column "local_path" = full path to each image.
    """

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3), # Normalize for RGB images
    ])

    clients_data = []

    # Note: clients numbered 1.....NUM_CLIENTS
    for cid in range(1, NUM_CLIENTS + 1):
        csv_path = os.path.join(base_dir, f"client{cid}_train_clean.csv")
        print(f"[INFO] Loading client {cid} CSV from: {csv_path}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing dataset file: {csv_path}")

        dataset = CustomDataset(
            csv_file=csv_path,
            label_col=label_col,
            transform=transform,
            path_col="local_path",   # your CSV has this column
            client_dir=None,         # not needed since local_path is absolute
        )
        clients_data.append(dataset)

    print(f"[INFO] Loaded {len(clients_data)} client datasets.")
    return clients_data


