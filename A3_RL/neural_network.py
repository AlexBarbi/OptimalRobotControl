import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
import l4casadi as l4

class NeuralNetwork(nn.Module):
    """ A simple feedforward neural network. """
    def __init__(self, input_size, hidden_size, output_size, activation=nn.Tanh(), ub=None):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, output_size),
            activation,
        )
        self.ub = ub if ub is not None else 1 
        self.initialize_weights()

    def forward(self, x):
        # Handle CasADi column vector input (nx, 1) -> (1, nx)
        if x.dim() == 2 and x.shape[1] == 1 and x.shape[0] == self.linear_stack[0].in_features:
            x = x.T
        out = self.linear_stack(x) * self.ub
        return out
   
    def initialize_weights(self):
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    
    def create_casadi_function(self):
        """
        Create a l4casadi function that can be used in CasADi graphs.
        """
        # Ensure model is in eval mode
        self.eval()
        
        # Create l4casadi model
        # Assuming input dimension is 2*nq (from main.py config mainly, but inferred here)
        # We need to know the input shape.
        # The input_size is saved in self.linear_stack[0].in_features
        input_dim = self.linear_stack[0].in_features
        
        l4_model = l4.L4CasADi(self, device='cpu', name='terminal_cost')
        
        return l4_model

def train_network(x_data, y_data, batch_size=32, epochs=10000, lr=1e-3):
    """
    Funzione per allenare la rete. Può essere chiamata dal main script.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Converti numpy -> tensor
    X = torch.tensor(x_data, dtype=torch.float32)
    Y = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)

    # Dataset
    dataset = TensorDataset(X, Y)
    test_split = 0.2
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Init Model
    input_dim = X.shape[1]
    output_dim = 1
    max_cost = Y.max().item()
    ub_val = max_cost * 1.2
    
    model = NeuralNetwork(input_dim, 64, output_dim, ub=ub_val).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 5. Training Loop
    train_losses = []
    test_losses = []

    best_model = None
    best_loss  = 1e10

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_loss / train_size
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_test_loss += loss.item() * inputs.size(0)
        
        epoch_test_loss = running_test_loss / test_size
        test_losses.append(epoch_test_loss)

        if epoch_test_loss < best_loss:
            best_loss = epoch_test_loss
            best_model = model

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_test_loss:.4f}")

    # 6. Save the Model
    model = best_model
    # Saving both the state dictionary (weights) and the full model for easier loading
    # Ensure model directory exists and save inside it
    os.makedirs('model', exist_ok=True)
    save_path = os.path.join('model', 'model.pt')
    torch.save({'model': model.state_dict(), 'ub': ub_val}, save_path)
    print(f"Model with {best_loss} saved to '{save_path}'")
    
    plt.figure(figsize=(12, 5))
        
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training History')
    plt.legend()

    # Prediction Accuracy
    plt.subplot(1, 2, 2)
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            preds = model(inputs).cpu()
            all_preds.append(preds)
            all_targets.append(targets)
    
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    plt.scatter(all_targets, all_preds, alpha=0.5, s=10)
    # Ideal line
    min_val, max_val = all_targets.min(), all_targets.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    
    plt.xlabel('Ground Truth Cost')
    plt.ylabel('Predicted Cost')
    plt.title('Prediction Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model