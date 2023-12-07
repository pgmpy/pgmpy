import torch

from pgmpy.utils import optimize


def l2_loss(w, data):
    """
    The loss function to use for optimizing the structure.
    """
    nsamples = data.shape[0]
    residual = data - data @ w
    return (0.5 / nsamples) * (residual ** 2).sum()


class NoTears(StructureEstimator):
    """
    NoTears algorithm is a score based optimization algorithm in which we optimize the loss function with the constraint that the structure
    remains a DAG.
    """

    def h(self, w):
        """
        Function to maintain the adjacency matrix to be a DAG.
        """
        nvars = self.data.shape[1]
        M = torch.eye(nvars) + (w * w / nvars)
        E = torch.matrix_power(M, nvars - 1)
        return E.transpose(0, 1) * w * 2

    def opt_fn(self, params, loss_args):
        hw = self.h(params["w"])
        obj = (
            loss_args["loss_fn"](params["w"], self.data)
            + (loss_args["rho"] / 2) * (hw ** 2)
            + loss_args["alpha"] * hw
            + loss_args["lamb"] * params["w"].sum()
        )

    def estimate(
        alpha,
        lamb,
        max_iter,
        h_tol,
        rho_max,
        init_w=None,
        rho_max=1e16,
        loss_fn=l2_loss,
        threshold=0.01,
    ):
        """
        Estimates the structure from the data.

        Parameters
        ----------

        """
        nsamples, nvars = self.data.shape
        if init_w is None:
            w = torch.normal(mean=0, std=1, size=(nvars, nvars), requires_grad=True)
        else:
            w = torch.tensor(init_w, requires_grad=True)

        rho = 1.0
        for _ in range(max_iter):
            while rho < rho_max:
                params = optimize(
                    loss_fn=self.opt_fn,
                    params={"w": w},
                    loss_args={
                        "loss_fn": loss_fn,
                        "rho": rho,
                        "alpha": alpha,
                        "lamb": lamb,
                    },
                    max_iter=max_iter,
                    exit_delta=exit_delta,
                )
                if self.h(params["w"]) > 0.25 * self.h(w):
                    rho *= 10
                else:
                    break
                w = params["w"]

            alpha += rho * self.h(w)
            if (h <= h_tol) or (rho >= rho_max):
                break

        return np.where(w < threshold, 0, w)
