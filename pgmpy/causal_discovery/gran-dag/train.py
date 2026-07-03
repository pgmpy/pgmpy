import numpy as np
import torch
from dag_optim import compute_constraint

from pgmpy.base import DAG


class SimpleDataLoader:
    def __init__(self, data, normalize=False):
        """
        data: pandas DataFrame or numpy array (N, d)
        """
        if hasattr(data, "values"):
            data = data.values  # convert DataFrame → numpy

        self.data = torch.tensor(data, dtype=torch.float32)

        if normalize:
            self.mean = self.data.mean(0, keepdim=True)
            self.std = self.data.std(0, keepdim=True) + 1e-8
            self.data = (self.data - self.mean) / self.std

        self.num_samples = self.data.shape[0]

    def sample(self, batch_size):
        idx = np.random.choice(self.num_samples, batch_size, replace=False)
        return self.data[idx]


def _initialize_model(self, d):
    import torch
    from model import GraNDAGModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraNDAGModel(
        num_vars=d, hidden_dim=self.hidden_dim, num_layers=self.num_layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

    return model, optimizer, device


def _train(self, model, loader, device):

    mu = self.mu_init
    lamb = self.lambda_init

    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

    for step in range(self.num_train_iter):

        model.train()

        # ---- Sample batch ----
        x = loader.sample(self.batch_size).to(device)

        # ---- Forward (NLL) ----
        weights, biases, extra_params = model.get_parameters(mode="wbx")

        loss = -torch.mean(
            model.compute_log_likelihood(x, weights, biases, extra_params)
        )

        # ---- DAG constraint ----
        w_adj = model.get_w_adj()
        h = compute_constraint(model, w_adj)

        # ---- Augmented Lagrangian ----
        aug_lagrangian = loss + 0.5 * mu * h**2 + lamb * h

        # ---- Optimization ----
        optimizer.zero_grad()
        aug_lagrangian.backward()
        optimizer.step()

        # ---- Edge clamping (optional but useful) ----
        if self.edge_clamp_range > 0:
            with torch.no_grad():
                mask = (w_adj > self.edge_clamp_range).float()
                model.adjacency *= mask

        # ---- Update dual variables (IMPORTANT) ----
        if step % self.dual_update_freq == 0:
            h_val = h.item()

            if h_val > self.h_tol:
                mu *= self.mu_mult_factor

            lamb += mu * h_val

        # ---- (Optional) convergence check ----
        if h.item() < self.h_tol:
            break


class GraNDAG:

    def __init__(self, data):
        self.data = data
        self.variables = list(data.columns)

    def estimate(self):

        loader = SimpleDataLoader(self.data)

        d = len(self.variables)

        model, optimizer, device = self._initialize_model(d)

        self._train(model, loader, device)

        W = model.get_w_adj().detach().cpu().numpy()

        return self._to_pgmpy_dag(W)

    def _to_pgmpy_dag(self, W):
        edges = []
        d = len(self.variables)

        for i in range(d):
            for j in range(d):
                if W[i, j] > 0:
                    edges.append((self.variables[j], self.variables[i]))

        return DAG(edges)
