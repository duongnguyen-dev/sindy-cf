import torch
from torch import nn
from torch.nn import Module, init

class NonlinearRegressionModel(Module):
    def __init__(self):
        super().__init__()

        diagonal_matrix = torch.eye(3)
        xavier_matrix = torch.empty(3, 4)
        init.xavier_normal_(xavier_matrix)

        combined_matrix = torch.cat([diagonal_matrix, xavier_matrix], dim=1)
        self.weights = nn.Parameter(
            combined_matrix
        )
        
        # Freeze the first three weights
        nontrainable_mask = torch.zeros(3, 3)
        trainable_mask = torch.ones(3, 4)
        mask = torch.cat([nontrainable_mask, trainable_mask], dim=1)

        self.register_buffer("mask", mask)

    def forward(self, x):
        masked_weights = self.weights * self.mask
        return x @ masked_weights.T
