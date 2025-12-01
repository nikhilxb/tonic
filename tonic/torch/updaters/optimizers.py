import typing as T

import torch


FLOAT_EPSILON = 1e-8


def flat_concat(xs: T.Sequence[torch.Tensor]) -> torch.Tensor:
  """Concatenate tensors into a single flat vector.
  
  Args:
    xs: Sequence of tensors to concatenate, e.g. [(dim1,), (dim2, dim3), ...].
    
  Returns:
    Single 1D tensor [sum(dims)] containing all values from input tensors.
  """
  return torch.cat([torch.reshape(x, (-1,)) for x in xs], dim=0)  # [sum_dims]


def assign_params_from_flat(
  new_params: torch.Tensor,
  params: list[torch.nn.Parameter],
) -> None:
  """Assign values from a flat vector to a list of parameters.
  
  Args:
    new_params: Flat 1D tensor containing new parameter values.
    params: List of parameters to update. Must have same total size as new_params.
  """
  with torch.no_grad():  # Parameter assignment doesn't need gradient tracking.
    splits = torch.split(new_params, [p.numel() for p in params])
    new_params_list = [torch.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
    for p, p_new in zip(params, new_params_list):
      p.data.copy_(p_new)


class ConjugateGradient:
  """Conjugate gradient optimizer for TRPO-style natural gradient updates."""

  def __init__(
    self,
    conjugate_gradient_steps: int = 10,
    damping_coefficient: float = 0.1,
    constraint_threshold: float = 0.01,
    backtrack_steps: int = 10,
    backtrack_coefficient: float = 0.8,
  ):
    """Initialize the conjugate gradient optimizer.
    
    Args:
      conjugate_gradient_steps: Maximum iterations for CG algorithm.
      damping_coefficient: Damping factor for Fisher-vector products (numerical stability).
      constraint_threshold: Maximum KL divergence constraint.
      backtrack_steps: Maximum line search backtracking steps.
      backtrack_coefficient: Multiplicative factor for line search (<1).
    """
    self.conjugate_gradient_steps = conjugate_gradient_steps
    self.damping_coefficient = damping_coefficient
    self.constraint_threshold = constraint_threshold
    self.backtrack_steps = backtrack_steps
    self.backtrack_coefficient = backtrack_coefficient

  def optimize(
    self,
    loss_function: T.Callable[[], torch.Tensor],
    constraint_function: T.Callable[[], torch.Tensor],
    variables: list[torch.nn.Parameter],
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform natural gradient optimization with KL constraint.
    
    Uses conjugate gradient to solve for the natural gradient direction,
    then performs line search to satisfy KL constraint.
    
    Args:
      loss_function: Callable returning the loss to minimize.
      constraint_function: Callable returning the KL divergence constraint.
      variables: List of parameters to optimize.
      
    Returns:
      Tuple of (final KL divergence, final loss, number of backtracking steps).
    """

    def _hx(x: torch.Tensor) -> torch.Tensor:
      """Compute Hessian-vector product (Fisher-vector product).
      
      Args:
        x: Input vector [param_dim].
        
      Returns:
        Hessian-vector product [param_dim].
      """
      f = constraint_function()  # []
      gradient_1 = flat_concat(torch.autograd.grad(f, variables, create_graph=True))  # [param_dim]
      x = torch.as_tensor(x)  # [param_dim]
      y = (gradient_1 * x).sum()  # [] - scalar for second derivative
      gradient_2 = flat_concat(torch.autograd.grad(y, variables))  # [param_dim]

      if self.damping_coefficient > 0:
        gradient_2 = gradient_2 + self.damping_coefficient * x  # [param_dim]

      return gradient_2  # [param_dim]

    def _cg(b: torch.Tensor) -> torch.Tensor | None:
      """Solve Ax = b using conjugate gradient.
      
      Pure PyTorch implementation of CG algorithm for solving linear systems.
      """
      with torch.no_grad():  # CG is numerical linear algebra, no gradients needed.
        x = torch.zeros_like(b)  # [param_dim]
        r = b.clone()  # [param_dim] - residual
        p = r.clone()  # [param_dim] - search direction
        r_dot_old = torch.dot(r, r)  # [] - r^T r
        if r_dot_old == 0:
          return None

        for _ in range(self.conjugate_gradient_steps):
          z = _hx(p)  # [param_dim] - Hessian-vector product
          alpha = r_dot_old / (torch.dot(p, z) + FLOAT_EPSILON)  # [] - p^T z
          x = x + alpha * p  # Update solution
          r = r - alpha * z  # Update residual
          r_dot_new = torch.dot(r, r)  # [] - r^T r
          p = r + (r_dot_new / r_dot_old) * p  # Update search direction
          r_dot_old = r_dot_new
        return x

    def _update(
      alpha: float,
      conjugate_gradient: torch.Tensor,
      step: float,
      start_variables: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
      """Update parameters and evaluate constraint and loss.
      
      Args:
        alpha: Maximum step size [].
        conjugate_gradient: Natural gradient direction [param_dim].
        step: Line search coefficient [].
        start_variables: Original parameter values [param_dim].
        
      Returns:
        Tuple of (KL divergence [], loss []).
      """
      with torch.no_grad():  # Line search evaluation doesn't need gradient tracking.
        conjugate_gradient = torch.as_tensor(conjugate_gradient)  # [param_dim]
        new_variables = start_variables - alpha * conjugate_gradient * step  # [param_dim]
        assign_params_from_flat(new_variables, variables)
      constraint = constraint_function()  # []
      loss = loss_function()  # []
      return constraint.detach(), loss.detach()

    start_variables = flat_concat(variables)  # [param_dim]

    # Zero out any existing gradients.
    for var in variables:
      if var.grad:
        var.grad.data.zero_()

    loss = loss_function()  # []
    grad = flat_concat(torch.autograd.grad(loss, variables))  # [param_dim]
    start_loss = loss.detach()  # []

    # Solve for natural gradient direction using CG.
    conjugate_gradient = _cg(grad)  # [param_dim]
    if conjugate_gradient is None:
      # CG failed, return zeros.
      constraint = torch.as_tensor(0.0, dtype=torch.float32)  # []
      loss = torch.as_tensor(0.0, dtype=torch.float32)  # []
      steps = torch.as_tensor(0, dtype=torch.int32)  # []
      return constraint, loss, steps

    # Compute maximum step size satisfying KL constraint.
    with torch.no_grad():  # Step size computation is purely numerical.
      hx = _hx(conjugate_gradient)  # [param_dim]
      alpha = torch.sqrt(
        2 * self.constraint_threshold / (conjugate_gradient * hx).sum() + FLOAT_EPSILON
      ).item()  # [] - scalar step size

    # Line search: backtrack until KL and loss improvement satisfied.
    if self.backtrack_steps is None or self.backtrack_coefficient is None:
      constraint, loss = _update(alpha, conjugate_gradient, 1, start_variables)  # [], []
      steps = torch.as_tensor(0, dtype=torch.int32)  # []
      return constraint, loss, steps

    for i in range(self.backtrack_steps):
      constraint, loss = _update(
        alpha, conjugate_gradient, self.backtrack_coefficient**i, start_variables
      )  # [], []

      if constraint.item() <= self.constraint_threshold and loss.item() <= start_loss.item():
        break

      if i == self.backtrack_steps - 1:
        # Line search failed, revert to original parameters.
        constraint, loss = _update(alpha, conjugate_gradient, 0, start_variables)  # [], []
        i = self.backtrack_steps

    return constraint, loss, torch.as_tensor(i + 1, dtype=torch.int32)  # [], [], []
