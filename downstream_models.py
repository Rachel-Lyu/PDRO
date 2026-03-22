import torch
import torch.nn as nn
import torch.nn.functional as F

# DOWNSTREAM MODELS
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 activations=None):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        if activations is not None:
            for i in range(len(activations)):
                if activations[i] == 'relu':
                    activations[i] = F.relu
                elif activations[i] == 'sigmoid':
                    activations[i] = F.sigmoid
                elif activations[i] == 'tanh':
                    activations[i] = F.tanh
                elif activations[i] == 'softmax':
                    activations[i] = F.softmax
                else:
                    raise ValueError(f"Unsupported activation function: {activations[i]}")
        else:
            self.activations = [F.relu, F.relu]

    def forward(self, x):
        x = self.fc1(x)
        x = self.activations[0](x)
        x = self.fc2(x)
        x = self.activations[1](x)
        return x
    
    def train_epoch(self, data_loader, optimizer, criterion):
        self.train()
        total_loss = 0
        for batch in data_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)
    
    def evaluate(self, data_loader, criterion):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                inputs, targets = batch
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(data_loader)
    
# UTILITY FUNCTIONS
