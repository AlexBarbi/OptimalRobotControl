"""Dataset generation, bootstrapping and fine-tuning helpers."""
import numpy as np
import torch
import multiprocessing
import time
from ocp import solve_single_ocp, solve_single_ocp_return_terminal, solve_single_ocp_with_terminal_return_terminal
from config import nq, NUM_SAMPLES


def generate_random_state():
    q_min, q_max = -np.pi, np.pi
    dq_min, dq_max = -8.0, 8.0
    q_rand = np.random.uniform(q_min, q_max, nq)
    dq_rand = np.random.uniform(dq_min, dq_max, nq)
    return np.concatenate([q_rand, dq_rand])


def generate_grid_states(num_samples, nq_local, q_lims, dq_lims):
    nx = 2 * nq_local
    points_per_dim = int(num_samples ** (1 / nx))
    q_ranges = [np.linspace(q_lims[0], q_lims[1], points_per_dim) for _ in range(nq_local)]
    dq_ranges = [np.linspace(dq_lims[0], dq_lims[1], points_per_dim) for _ in range(nq_local)]
    all_ranges = q_ranges + dq_ranges
    mesh = np.meshgrid(*all_ranges)
    states_grid = np.vstack([m.flatten() for m in mesh]).T
    return states_grid


def collect_terminal_dataset(num_samples, M, N_long, grid=False, seed=None, processes=None):
    if seed is not None:
        np.random.seed(seed)

    if not grid:
        initial_states = [generate_random_state() for _ in range(num_samples)]
    else:
        initial_states = [generate_random_state() for _ in range(num_samples)]

    procs = processes or multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=procs) as pool:
        results = pool.map(solve_single_ocp_return_terminal, initial_states)

    valid = [r for r in results if r is not None]
    xM_list = [r[2] for r in valid]

    with multiprocessing.Pool(processes=procs) as pool:
        VN_results = pool.map(lambda args: solve_single_ocp(args, N=N_long), xM_list)

    X_boot = []
    Y_boot = []
    for x, res in zip(xM_list, VN_results):
        if res is not None:
            X_boot.append(x)
            Y_boot.append(res[1])

    return np.array(X_boot), np.array(Y_boot)


def fine_tune_model(model, X_boot, Y_boot, epochs=200, batch_size=64, lr=1e-4, save_path='model_finetuned.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    X = torch.tensor(X_boot, dtype=torch.float32)
    Y = torch.tensor(Y_boot, dtype=torch.float32).view(-1, 1)

    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        running = 0.0
        count = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            count += xb.size(0)
        epoch_loss = running / count
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {'model': model.state_dict(), 'ub': model.ub}
        if (epoch + 1) % 50 == 0:
            print(f"Finetune Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    torch.save(best_state, save_path)
    model.load_state_dict(best_state['model'])
    model.to(device)
    model.eval()
    print(f"Fine-tuned model saved to {save_path} with best loss {best_loss:.4f}")
    return model