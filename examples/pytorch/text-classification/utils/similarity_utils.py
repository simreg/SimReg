from typing import Optional
import torch
from torch import Tensor
from torch.linalg import norm
from functools import partial


def _check_shape_equal(x: Tensor, y: Tensor, dim: int):
    if x.size(dim) != y.size(dim):
        raise ValueError(f'x.size({dim}) == y.size({dim}) is expected, but got {x.size(dim)=}, {y.size(dim)=} instead.')

def _zero_mean(input: Tensor, dim: int) -> Tensor:
    return input - input.mean(dim=dim, keepdim=True)

def _svd(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # torch.svd style
    U, S, Vh = torch.linalg.svd(input, full_matrices=False)
    V = Vh.transpose(-2, -1)
    return U, S, V

def _matrix_normalize(input: Tensor, dim: int) -> Tensor:
    """
    Center and normalize according to the forbenius norm of the centered data.
    Note:
        - this does not create standardized random variables in a random vectors.
    ref:
        - https://stats.stackexchange.com/questions/544812/how-should-one-normalize-activations-of-batches-before-passing-them-through-a-si
    :param input:
    :param dim:
    :return:
    """
    X_centered: Tensor = _zero_mean(input, dim=dim)
    X_star: Tensor = X_centered / norm(X_centered, "fro")
    return X_star


def cca_by_svd(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """ CCA using only SVD.
    For more details, check Press 2011 "Canonical Correlation Clarified by Singular Value Decomposition"
    Args:
        x: input tensor of Shape DxH
        y: input tensor of shape DxW
    Returns: x-side coefficients, y-side coefficients, diagonal
    """

    # torch.svd(x)[1] is vector
    u_1, s_1, v_1 = _svd(x)
    u_2, s_2, v_2 = _svd(y)
    uu = u_1.t() @ u_2
    u, diag, v = _svd(uu)
    # a @ (1 / s_1).diag() @ u, without creating s_1.diag()
    a = v_1 @ (1 / s_1[:, None] * u)
    b = v_2 @ (1 / s_2[:, None] * v)
    return a, b, diag


def cca_by_qr(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """ CCA using QR and SVD.
    For more details, check Press 2011 "Canonical Correlation Clarified by Singular Value Decomposition"
    Args:
        x: input tensor of Shape DxH
        y: input tensor of shape DxW
    Returns: x-side coefficients, y-side coefficients, diagonal
    """

    q_1, r_1 = torch.linalg.qr(x)
    q_2, r_2 = torch.linalg.qr(y)
    qq = q_1.t() @ q_2
    u, diag, v = _svd(qq)
    # a = r_1.inverse() @ u, but it is faster and more numerically stable
    a = torch.linalg.solve(r_1, u)
    b = torch.linalg.solve(r_2, v)
    return a, b, diag


def cca(x: Tensor, y: Tensor, backend: str) -> tuple[Tensor, Tensor, Tensor]:
    """ Compute CCA, Canonical Correlation Analysis
    Args:
        x: input tensor of Shape DxH
        y: input tensor of Shape DxW
        backend: svd or qr
    Returns: x-side coefficients, y-side coefficients, diagonal
    """

    _check_shape_equal(x, y, 0)

    if x.size(0) < x.size(1):
        raise ValueError(f'x.size(0) >= x.size(1) is expected, but got {x.size()=}.')

    if y.size(0) < y.size(1):
        raise ValueError(f'y.size(0) >= y.size(1) is expected, but got {y.size()=}.')

    if backend not in ('svd', 'qr'):
        raise ValueError(f'backend is svd or qr, but got {backend}')

    x = _zero_mean(x, dim=0)
    y = _zero_mean(y, dim=0)
    return cca_by_svd(x, y) if backend == 'svd' else cca_by_qr(x, y)



def pwcca_distance_choose_best_layer_matrix(x: Tensor, y: Tensor, backend: str, use_layer_matrix: Optional[str] = None,
                                            epsilon: float = 1e-10) -> Tensor:
    """ Projection Weighted CCA proposed in Marcos et al. 2018.
    ref:
        - https://github.com/moskomule/anatome/issues/30
    Args:
        x: input tensor of Shape NxD1, where it's recommended that N>Di
        y: input tensor of Shape NxD2, where it's recommended that N>Di
        backend: svd or qr
    Returns:
    """
    x = _zero_mean(x, dim=0)
    y = _zero_mean(y, dim=0)
    B, D1 = x.size()
    B2, D2 = y.size()
    assert B == B2
    C_ = min(D1, D2)
    a, b, diag = cca(x, y, backend)
    C = diag.size(0)
    assert (C == C_)
    assert a.size() == torch.Size([D1, C])
    assert diag.size() == torch.Size([C])
    assert b.size() == torch.Size([D2, C])
    if use_layer_matrix is None:
        # sigma_xx_approx = x
        # sigma_yy_approx = y
        sigma_xx_approx = x.T @ x
        sigma_yy_approx = y.T @ y
        x_diag = torch.diag(sigma_xx_approx.abs())
        y_diag = torch.diag(sigma_yy_approx.abs())
        x_idxs = (x_diag >= epsilon)
        y_idxs = (y_diag >= epsilon)
        use_layer_matrix: str = 'x' if x_idxs.sum() <= y_idxs.sum() else 'y'
    if use_layer_matrix == 'x':
        x_tilde = x @ a
        assert x_tilde.size() == torch.Size([B, C])
        x_tilde, _ = torch.linalg.qr(input=x_tilde)
        assert x_tilde.size() == torch.Size([B, C])
        alpha_tilde_dot_x_abs = (x_tilde.T @ x).abs_()
        assert alpha_tilde_dot_x_abs.size() == torch.Size([C, D1])
        alpha_tilde = alpha_tilde_dot_x_abs.sum(dim=1)
        assert alpha_tilde.size() == torch.Size([C])
    elif use_layer_matrix == 'y':
        y_tilde = y @ b
        assert y_tilde.size() == torch.Size([B, C])
        y_tilde, _ = torch.linalg.qr(input=y_tilde)
        assert y_tilde.size() == torch.Size([B, C])
        alpha_tilde_dot_y_abs = (y_tilde.T @ y).abs_()
        assert alpha_tilde_dot_y_abs.size() == torch.Size([C, D2])
        alpha_tilde = alpha_tilde_dot_y_abs.sum(dim=1)
        assert alpha_tilde.size() == torch.Size([C])
    else:
        raise ValueError(f"Invalid input: {use_layer_matrix=}")
    assert alpha_tilde.size() == torch.Size([C])
    alpha = alpha_tilde / alpha_tilde.sum()
    assert alpha_tilde.size() == torch.Size([C])
    # important, returning the similarity instead of distance (to minimize it) -- returning # instead of 1 - #
    return alpha @ diag


def orthogonal_procrustes_distance(x: Tensor, y: Tensor) -> Tensor:
    """ Orthogonal Procrustes distance used in Ding+21.
    Returns in dist interval [0, 1].
    Note:
        -  for a raw representation A we first subtract the mean value from each column, then divide
    by the Frobenius norm, to produce the normalized representation A* , used in all our dissimilarity computation.
        - see uutils.torch_uu.orthogonal_procrustes_distance to see my implementation
    Args:
        x: input tensor of Shape NxD1
        y: input tensor of Shape NxD2
    Returns:
    """
    # _check_shape_equal(x, y, 0)
    nuclear_norm = partial(torch.linalg.norm, ord="nuc")

    x = _matrix_normalize(x, dim=0)
    y = _matrix_normalize(y, dim=0)
    # note: ||x||_F = 1, ||y||_F = 1
    # - note this already outputs it between [0, 1] e.g. it's not 2 - 2 nuclear_norm(<x1, x2>) due to 0.5*d_proc(x, y)
    # important, returning the similarity instead of distance (to minimize it) -- returning # instead of 1 - #
    return nuclear_norm(x.t() @ y)
