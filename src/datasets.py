import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from helpers import compute_cutting_foce, compute_smoothed_diff

class CuttingForceDataset(Dataset):
    def __init__(self, csv_file: str, transform=None, target_transform=None):
        
        self.cutting_force_ds = pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.cutting_force_ds)

    def __getitem__(self, idx):
        sample = {
            "input":self.cutting_force_ds.iloc[:, 1:].values[idx],
            "output": None
        }

        if self.transform:
            sample["input"] = self.transform(sample["input"])
        
        if self.target_transform:
            sample["output"] = self.target_transform(sample["input"])

        return sample
    
def create_dataset(csv_file):
    # return CuttingForceDataset(csv_file, compute_cutting_foce, compute_smoothed_diff)
    return CuttingForceDataset(csv_file, compute_cutting_foce)

if __name__ == "__main__":
    ds = create_dataset("../data/ss3000_ap2.0_ft600_ad7.5_f50k.csv")
    for i, sample in enumerate(ds):
        print(sample)
        break