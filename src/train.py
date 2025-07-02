import os
import torch
from torch.utils.data import DataLoader
from torcheval.metrics import R2Score
from configs import *
from models.nonlinear_regression import NonlinearRegressionModel
from datasets import CuttingForceDataset
from transforms import get_transforms

def train(train_ds, val_ds, path):
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())

    model = NonlinearRegressionModel()

    loss_fn = torch.nn.L1Loss(reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)  # L2 via weight_decay
    l1_lambda = LAMBDA
    
    for epoch in range(EPOCHS):
        # Training phases
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            X_train, y_train = batch["input"].to(DEVICE), batch["output"].to(DEVICE)

            # forward pass
            y_pred = model(X_train)

            # Calculate loss function 
            l1_norm = sum(p.abs().sum() for name, p in model.named_parameters() if 'weight' in name)
            loss = loss_fn(y_pred, y_train) + l1_lambda * l1_norm

            # Optimizer zero grad
            optimizer.zero_grad()

            # backward pass
            loss.backward()
            total_train_loss += loss.item()

            # Perform optimizer
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)

        # Evaluation phase
        r2_metric = R2Score()
        model.eval()
        r2_metric.reset()
        total_eval_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                X_val, y_val = batch["input"].to(DEVICE), batch["output"].to(DEVICE)

                # Forward pass
                y_pred = model(X_val)

                # Calculate loss function 
                l1_norm = sum(p.abs().sum() for name, p in model.named_parameters() if 'weight' in name)
                loss = loss_fn(y_pred, y_val) + l1_lambda * l1_norm
                total_eval_loss = loss.item()

                r2_metric.update(y_pred, y_val)
    
        r2 = r2_metric.compute()
        avg_eval_loss = total_eval_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{EPOCHS}: ")
        print(f"Train loss = {avg_train_loss:.4f}, Eval loss = {avg_eval_loss:.4f}, R-squared = {r2:.4f}")

        torch.save(model.state_dict(), path)
        print("Save model successfully!")

if __name__ == "__main__":
    train_ds = CuttingForceDataset(csv_file="../data/train.csv", transform=get_transforms(mode="train"))
    val_ds = CuttingForceDataset(csv_file="../data/test.csv", transform=get_transforms(mode="train"))

    train(train_ds, val_ds, "../models/test.pth")