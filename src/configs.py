import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMBDA = 0.001
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
EPOCHS = 1
BATCH_SIZE = 32
DATASET_DIR = "../data/ss3000_ap2.0_ft600_ad10.0_f50k.csv"