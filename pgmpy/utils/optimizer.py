from math import isclose

from pgmpy.global_vars import logger


try:  # pragma: no cover
    import torch

    optim = torch.optim
except ImportError:  # pragma: no cover
    optim = None


def pinverse(t):
    """
    Computes the pseudo-inverse of a matrix using SVD.

    Parameters
    ----------
    t: torch.tensor
        The matrix whose inverse is to be calculated.

    Returns
    -------
    torch.tensor: Inverse of the matrix `t`.
    """
    u, s, v = t.svd()
    t_inv = v @ torch.diag(torch.where(s != 0, 1 / s, s)) @ u.t()
    return t_inv


def optimize(
    loss_fn, params={}, loss_args={}, opt="adam", max_iter=10000, exit_delta=1e-4
):
    """
    Generic function to optimize loss functions.

    Parameters
    ----------
    loss_fn: Function
        The function to optimize. It must return a torch.Tensor object.

    params: dict {str: torch.Tensor}
        The parameters which need to be optimized along with their initial values. The
        dictionary should be of the form: {variable name: initial value}

    loss_args: dict {str: torch.Tensor}
        Extra parameters which loss function needs to compute the loss.

    opt: str | Instance of torch.optim.Optimizer
        The optimizer to use. Should either be an instance of torch.optim or a str.
        When str is given initializes the optimizer with default parameters.

        If str the options are:
            1. Adadelta: Adadelta algorithm (Ref: https://arxiv.org/abs/1212.5701)
            2. Adagrad: Adagrad algorithm (Ref: http://jmlr.org/papers/v12/duchi11a.html)
            3. Adam: Adam algorithm (Ref: https://arxiv.org/abs/1412.6980)
            4. SparseAdam: Lazy version of Adam. Suitable for sparse tensors.
            5. Adamax: Adamax algorithm (variant of Adam based on infinity norm)
            6. ASGD: Averaged Stochastic Gradient Descent (Ref: https://dl.acm.org/citation.cfm?id=131098)
            7. LBFGS: L-BFGS Algorithm
            8. RMSprop: RMSprop Algorithm (Ref: https://arxiv.org/abs/1308.0850v5)
            9. Rprop: Resilient Backpropagation Algorithm
            10. SGD: Stochastic Gradient Descent.

    max_iter: int (default: 10000)
        The maximum number of iterations to run the optimization for.

    exit_delta: float
        The optimization exit criteria. When change in loss in an iteration is less than
        `exit_delta` the optimizer returns the values.

    Returns
    -------
    dict: The values that were given in params in the same format.

    Examples
    --------
    """
    # TODO: Add option to modify the optimizers.
    init_loss = float("inf")

    if isinstance(opt, str):
        opt_dict = {
            "adadelta": optim.Adadelta,
            "adagrad": optim.Adagrad,
            "adam": optim.Adam,
            "sparseadam": optim.SparseAdam,
            "adamax": optim.Adamax,
            "asgd": optim.ASGD,
            "lbfgs": optim.LBFGS,
            "rmsprop": optim.RMSprop,
            "rprop": optim.Rprop,
            "sgd": optim.SGD,
        }
        opt = opt_dict[opt.lower()](params.values())

    for t in range(max_iter):

        def closure():
            opt.zero_grad()
            loss = loss_fn(params, loss_args)
            loss.backward()
            return loss

        opt.step(closure=closure)

        if isclose(init_loss, closure().item(), abs_tol=exit_delta):
            logger.info(f"Converged after {t} iterations.")
            return params
        else:
            init_loss = closure().item()

    logger.info(
        f"Couldn't converge after {max_iter} iterations. Try increasing max_iter or change optimizer parameters"
    )
    return params
