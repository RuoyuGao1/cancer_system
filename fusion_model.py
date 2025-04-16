import torch
import torch.nn as nn

class SimpleGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return torch.relu(self.linear(x))

class CoxPH(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.linear(x)

class MultiOmicsModel(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.omics_encoders = nn.ModuleDict({
            "methylation": nn.Sequential(
                nn.Linear(input_dims["methylation"], 512),
                nn.ReLU()
            ),
            "rna": nn.Sequential(
                nn.Linear(input_dims["rna"], 512),
                nn.ReLU()
            ),
            "mutation": SimpleGCN(input_dims["mutation"], 512),
        })

        self.fusion = nn.Sequential(
            nn.Linear(512 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(64, 1 + 32)  # 1 = risk score, 32 = patient embedding

    def forward(self, x):
        encoded = [self.omics_encoders[k](x[k]) for k in x]
        fused = torch.cat(encoded, dim=1)
        output = self.output_layer(self.fusion(fused))
        return output
