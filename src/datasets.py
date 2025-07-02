import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transforms import get_transforms

class CuttingForceDataset(Dataset):
    def __init__(self, csv_file: str, transform=None):
        
        self.cutting_force_ds = pd.read_csv(csv_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.cutting_force_ds)

    def __getitem__(self, idx):
        sample = {
            "input":self.cutting_force_ds.iloc[:, :].values[idx],
            "output": None
        }
        print(self.transform(sample))
        if self.transform:
            sample = self.transform(sample)

        return sample
    
def create_dataset(csv_file):
    # return CuttingForceDataset(csv_file, compute_cutting_foce, compute_smoothed_diff)
    transforms = get_transforms()
    return CuttingForceDataset(csv_file, transforms)

if __name__ == "__main__":
    ds = create_dataset("../data/ss3000_ap2.0_ft600_ad10.0_f50k.csv")
    for i, sample in enumerate(ds):
        print(i, sample)