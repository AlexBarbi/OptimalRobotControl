import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import os
import time

from tqdm import tqdm
import l4casadi as l4

class NeuralNetwork(nn.Module):
    """ 
    A simple feedforward neural network for approximating the value function (terminal cost).
    
    Attributes:
        linear_stack (nn.Sequential): The stack of linear layers and activation functions.
        ub (float): Upper bound scaling factor for the output.
    """
    def __init__(self, input_size, hidden_size, output_size, activation=nn.ReLU(), ub=None):
        """
        Initializes the neural network.

        Args:
            input_size (int): Dimension of the input vector (state size).
            hidden_size (int): Number of neurons in the hidden layers.
            output_size (int): Dimension of the output vector (1 for scalar cost).
            activation (nn.Module, optional): Activation function. Defaults to nn.Tanh().
            ub (float, optional): Upper bound for output scaling. Defaults to 1 if None.
        """
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, output_size),
            nn.Softplus(),
        )
        self.ub = ub if ub is not None else 1 
        self.initialize_weights()

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The network output scaled by `ub`.
        """
        # Handle CasADi column vector input (nx, 1) -> (1, nx)
        # This check is crucial when passing CasADi expressions that might be dimensioned as column vectors
        if x.dim() == 2 and x.shape[1] == 1 and x.shape[0] == self.linear_stack[0].in_features:
            x = x.T
        out = self.linear_stack(x) # * self.ub
        return out
   
    def initialize_weights(self):
        """
        Initializes weights using Xavier Normal initialization and biases to zero.
        """
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    
    def create_casadi_function(self, name='terminal_cost'):
        """
        Creates a pure CasADi implementation of the network for direct integration into OCP.
        
        This avoids the overhead of calling out to PyTorch via l4casadi for simple networks
        by reconstructing the network graph using CasADi primitives. This is significantly faster
        for optimization solvers.

        Args:
            name (str): Name of the CasADi function.

        Returns:
            casadi.Function: A CasADi function representing the neural network.
        """
        import casadi as cs
        
        # Ensure model is on CPU for weight extraction
        self.cpu()
        self.eval()

        input_dim = self.linear_stack[0].in_features
        x = cs.SX.sym('x', input_dim, 1)
        y = x

        # Reconstruct the feedforward pass using CasADi operations
        with torch.no_grad():
            for layer in self.linear_stack:
                if isinstance(layer, nn.Linear):
                    # PyTorch Linear stores weight as (out_features, in_features)
                    # We compute y = W @ x + b
                    W = layer.weight.detach().numpy()
                    b = layer.bias.detach().numpy().reshape(-1, 1)
                    
                    y = cs.mtimes(cs.DM(W), y) + cs.DM(b)
                elif isinstance(layer, nn.Tanh):
                    y = cs.tanh(y)
                elif isinstance(layer, nn.ReLU):
                    eps = 1e-4
                    y = (y + cs.sqrt(y**2 + eps)) / 2.0
                elif isinstance(layer, nn.Softplus):
                    y = cs.if_else(y > 20, y, cs.log(1 + cs.exp(y)))
        
        # Apply scaling
        # y = y * self.ub

        # Return casadi function: f(x) = y
        return cs.Function(name, [x], [y])

def train_network(x_data, y_data, batch_size=32, epochs=50000, lr=1e-4, save_dir='model_double', patience=100):
    """
    Trains the neural network to approximate the value function.

    Args:
        x_data (np.ndarray): Input data (states).
        y_data (np.ndarray): Target data (optimal costs).
        batch_size (int, optional): Batch size for training. Defaults to 32.
        epochs (int, optional): Number of training epochs. Defaults to 500.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        save_dir (str, optional): Directory to save the trained model. Defaults to 'model_double'.
        patience (int, optional): Number of epochs to wait for improvement before early stopping. Defaults to 50.

    Returns:
        NeuralNetwork: The trained PyTorch model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Convert numpy arrays to PyTorch tensors
    X = torch.tensor(x_data, dtype=torch.float32)
    Y = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)

    # Prepare Dataset and DataLoader
    dataset = TensorDataset(X, Y)
    test_split = 0.2
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Initialize Model parameters
    input_dim = X.shape[1]
    output_dim = 1
    max_cost = Y.max().item()
    # ub_val = max_cost * 1.0 # Heuristic scaling factor
    ub_val = 1.0
    
    model = NeuralNetwork(input_dim, 128, output_dim, ub=ub_val).to(device)
    
    # Loss function and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    train_losses = []
    test_losses = []

    best_model_state = None
    best_loss  = 1e10
    patience_counter = 0

    start = time.time()
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        # Training phase
        model.train()
        running_loss = 0.0
        
        # Wrap train_loader with tqdm for batch-level progress
        total_samples = 0
        batch_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for inputs, targets in batch_loop:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)  # Track actual sample count
            
        epoch_train_loss = running_loss / total_samples  # Use actual count

        # Validation phase
        model.eval()
        running_test_loss = 0.0
        total_test_samples = 0  # ADD THIS
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_test_loss += loss.item() * inputs.size(0)
                total_test_samples += inputs.size(0)  # ADD THIS

        epoch_test_loss = running_test_loss / total_test_samples  # Use actual count
        test_losses.append(epoch_test_loss)

        # Save best model logic
        if epoch_test_loss < best_loss:
            best_loss = epoch_test_loss
            best_model_state = model.state_dict().copy()  # Create explicit copy
            patience_counter = 0
            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_test_loss:.4f} *")
        else:
            patience_counter += 1
            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_test_loss:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                print(f"Best validation loss: {best_loss:.4f}")
                break

    end = time.time()
    print(f"\nTraining completed in {end - start:.2f} seconds")
    print(f"Final train loss: {epoch_train_loss:.4f}, Final val loss: {epoch_test_loss:.4f}")
    print(f"Best validation loss: {best_loss:.4f}")
    
    # Save the best model found
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    # Save both model state and scaling factor
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'model.pt')
    torch.save({'model': model.state_dict(), 'ub': ub_val}, save_path)
    print(f"Model with val loss {best_loss:.5f} saved to '{save_path}'")
    
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