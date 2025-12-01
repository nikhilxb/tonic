import torch


def lambda_returns(
  values: torch.Tensor,
  next_values: torch.Tensor,
  rewards: torch.Tensor,
  resets: torch.Tensor,
  terminations: torch.Tensor,
  discount_factor: float,
  trace_decay: float,
) -> torch.Tensor:
  """Calculate lambda-returns using GAE (Generalized Advantage Estimation).
  
  Computes n-step returns with eligibility traces, blending TD(0) and Monte Carlo
  estimates. The trace_decay parameter (lambda) controls the bias-variance tradeoff.
  
  Args:
    values: Value estimates for current states, shape [num_steps, num_envs].
    next_values: Value estimates for next states, shape [num_steps, num_envs].
    rewards: Rewards received, shape [num_steps, num_envs].
    resets: Episode reset flags (1 if reset, 0 otherwise), shape [num_steps, num_envs].
    terminations: Episode termination flags (1 if done, 0 otherwise), shape [num_steps, num_envs].
    discount_factor: Discount factor (gamma) for future rewards.
    trace_decay: Eligibility trace decay (lambda) for return computation.
    
  Returns:
    Lambda-returns for each timestep, shape [num_steps, num_envs].
  """
  returns = torch.zeros_like(values)  # [num_steps, num_envs]
  last_returns = next_values[-1]  # [num_envs]
  
  for t in reversed(range(len(rewards))):
    # Blend TD(0) bootstrap with previous lambda-return.
    bootstrap = (1 - trace_decay) * next_values[t] + trace_decay * last_returns
    # Reset bootstrap to current value on episode resets (timeouts).
    bootstrap *= (1 - resets[t])
    bootstrap += resets[t] * next_values[t]
    # Zero out bootstrap on true terminations (no future value).
    bootstrap *= (1 - terminations[t])
    # Compute return: r_t + gamma * bootstrap.
    returns[t] = last_returns = rewards[t] + discount_factor * bootstrap
  
  return returns


def flatten_batch(batch: torch.Tensor) -> torch.Tensor:
  """Flatten time and environment dimensions into a single batch dimension.
  
  Converts buffers from [num_steps, num_envs, ...] to [num_steps * num_envs, ...]
  for batched training.
  
  Args:
    batch: Tensor with shape [num_steps, num_envs, ...].
    
  Returns:
    Flattened tensor with shape [num_steps * num_envs, ...].
  """
  shape = batch.shape  # [num_steps, num_envs, ...]
  new_shape = (shape[0] * shape[1],) + shape[2:]  # [num_steps * num_envs, ...]
  return batch.reshape(new_shape)
