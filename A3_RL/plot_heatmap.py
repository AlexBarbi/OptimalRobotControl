import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import config
from neural_network import NeuralNetwork

from config import PENDULUM, VELOCITY_LIMIT, ROBOT_TYPE

def plot_heatmap():
    # Helper to enforce 'double' configuration in case config was loaded differently (though I changed the file)
    # The config module executes on import, so changing the file before running this script is key.
    
    if PENDULUM == 'single_pendulum':
        model_path = os.path.join('model_single', 'model.pt')
    else:
        model_path = os.path.join('model_double', 'model.pt')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print(f"Please train the model first by running main.py (ensure ROBOT_TYPE='{ROBOT_TYPE}' in config.py).")
        return

    print(f"Loading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    # config.NX should be 4 for double pendulum
    input_size = config.NX
    hidden_size = config.HIDDEN_SIZE
    output_size = 1
    
    # 'ub' might be in checkpoint or we default to 1.0 (from training code logic)
    ub = checkpoint.get('ub', 1.0)
    
    model = NeuralNetwork(input_size, hidden_size, output_size, ub=ub)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    if PENDULUM == 'single_pendulum':
        print("Generating heatmap for State Space (q in [-pi, pi], dq in [-limit, limit])...")
    else:
        print("Generating heatmap for Zero Velocity slice (q1, q2 in [-pi, pi], dq1=dq2=0)...")

    # Grid generation
    resolution = 100
    q_range = np.linspace(-np.pi, np.pi, resolution)
    
    if PENDULUM == 'single_pendulum':
        dq_range = np.linspace(-VELOCITY_LIMIT[0], VELOCITY_LIMIT[0], resolution)
        Q1, Q2 = np.meshgrid(q_range, dq_range)
    else:
        Q1, Q2 = np.meshgrid(q_range, q_range)
    
    # Flatten for batch processing
    q1_flat = Q1.flatten()
    q2_flat = Q2.flatten()
    zeros = np.zeros_like(q1_flat)
    zeros = np.ones_like(q1_flat) * 10.0
    zeros2 = np.ones_like(q2_flat) * 10.0
    
    # Create input tensor (Batch, NX)
    # State order typically: q1, q2, dq1, dq2
    
    if PENDULUM == 'single_pendulum':
        # exact match for single pendulum state: [q, dq]
        inputs_np = np.stack([q1_flat, q2_flat], axis=1)
    else:
        # double pendulum: [q1, q2, dq1=0, dq2=0]
        inputs_np = np.stack([q1_flat, q2_flat, zeros, zeros2], axis=1)

    inputs_tensor = torch.tensor(inputs_np, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        values = model(inputs_tensor)
        
    values_np = values.cpu().numpy().reshape(resolution, resolution)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(Q1, Q2, values_np, levels=100, cmap='viridis')
    plt.colorbar(label='Value (Terminal Cost)')
    plt.scatter(0, 0, color='red', marker='x', s=50, label='Goal State')
    if PENDULUM == 'single_pendulum':
        plt.xlabel('q (rad)')
        plt.ylabel('dq (rad/s)')
        plt.title('Value Function Heatmap (Single Pendulum)')
        output_file = f'model_single/heatmap.png'
    else:
        plt.xlabel('q1 (rad)')
        plt.ylabel('q2 (rad)')
        plt.title(f'Value Function Heatmap (Double Pendulum) - dq1={zeros[0]}, dq2={zeros2[0]}')
        output_file = f'model_double/heatmap.png'
    
    plt.savefig(output_file)
    print(f"Heatmap saved to {output_file}")
    # plt.show() # Uncomment if running in a GUI environment

if __name__ == "__main__":
    plot_heatmap()
