import config

import casadi as cs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import os
import time

from tqdm import tqdm

class NeuralNetwork(nn.Module):
    """ 
    A simple feedforward neural network for approximating the value function (terminal cost).
    """
    def __init__(self, input_size, hidden_size, output_size, activation=nn.ReLU(), ub=None):
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
        if x.dim() == 2 and x.shape[1] == 1 and x.shape[0] == self.linear_stack[0].in_features:
            x = x.T
        out = self.linear_stack(x) 
        return out
   
    def initialize_weights(self):
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def create_casadi_function(self, name='terminal_cost'): 
        self.cpu()
        self.eval()

        input_dim = self.linear_stack[0].in_features
        x = cs.SX.sym('x', input_dim, 1)
        y = x

        with torch.no_grad():
            for layer in self.linear_stack:
                if isinstance(layer, nn.Linear):
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
        return cs.Function(name, [x], [y])

def train_network(x_data, y_data, batch_size=32, epochs=10000, lr=1e-4, save_dir='model_double', patience=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    X = torch.tensor(x_data, dtype=torch.float32)
    Y = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X, Y)
    test_split = 0.2
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    input_dim = X.shape[1]
    output_dim = 1
    ub_val = 1.0
    
    model = NeuralNetwork(input_dim, config.HIDDEN_SIZE, output_dim, ub=ub_val).to(device)
    
    # Loss
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    
    # Scheduler
    scheduler_patience = max(5, patience // 3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=scheduler_patience, 
        verbose=True,
        min_lr=1e-7
    )

    train_losses = []
    test_losses = []

    best_model_state = None
    best_loss = float('inf') 
    patience_counter = 0

    start = time.time()
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        # Training
        model.train()
        running_loss = 0.0
        total_samples = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            if torch.isnan(loss):
                print(f"\nERROR: Loss is NaN at epoch {epoch+1}")
                return model

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
        epoch_train_loss = running_loss / total_samples

        # Validation
        model.eval()
        running_test_loss = 0.0
        total_test_samples = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_test_loss += loss.item() * inputs.size(0)
                total_test_samples += inputs.size(0)

        epoch_test_loss = running_test_loss / total_test_samples
        test_losses.append(epoch_test_loss)
        
        # Step scheduler
        scheduler.step(epoch_test_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Early Stopping Logic
        if epoch_test_loss < best_loss:
            best_loss = epoch_test_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best validation loss: {best_loss:.4f}")
                break

    end = time.time()
    print(f"\nTraining completed in {end - start:.2f} seconds")
    print(f"Final train loss: {epoch_train_loss:.4f}, Final val loss: {epoch_test_loss:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'model.pt')
    torch.save({'model': model.state_dict(), 'ub': ub_val}, save_path)
    print(f"Model saved to '{save_path}'")
    
    # Plotting
    try:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Validation Loss')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.title(f'Training History (Training Time: {end - start:.2f} s)')
        plt.legend()

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
        min_val, max_val = all_targets.min(), all_targets.max()
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')
        plt.title('Accuracy')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_plots.png'))
        plt.show()
    except Exception as e:
        print(f"Errore durante il plotting: {e}")
    
    return model