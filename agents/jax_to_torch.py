#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#This file correct a bug in the jaxtotorchwrapper of the current brax version

from typing import Optional, Union

from brax.envs import wrappers
# NOTE: The following line will emit a warning and raise ImportError if `torch`
# isn't available.
import gym
from collections import abc
import functools
from typing import Any, Dict, Union
import warnings

from jax._src import dlpack as jax_dlpack
from jax.interpreters.xla import DeviceArray

try:
  # pylint:disable=g-import-not-at-top
  import torch
  from torch.utils import dlpack as torch_dlpack
except ImportError:
  warnings.warn(
      "brax.io.torch requires PyTorch. Please run `pip install torch` to use "
      "functions from this module.")
  raise

Device = Union[str, torch.device]


@functools.singledispatch
def torch_to_jax(value: Any) -> Any:
  """Converts values to JAX tensors."""
  # Don't do anything by default, and when a handler is registered for this type
  # of value, it gets used to convert it to a Jax DeviceArray.
  # NOTE: The alternative would be to raise an error when an unsupported value
  # is encountered:
  # raise NotImplementedError(f"Cannot convert {v} to a Jax tensor")
  return value


@torch_to_jax.register(torch.Tensor)
def _tensor_to_jax(value: torch.Tensor) -> DeviceArray:
  """Converts a PyTorch Tensor into a Jax DeviceArray."""
  tensor = torch_dlpack.to_dlpack(value)
  tensor = jax_dlpack.from_dlpack(tensor)
  return tensor


@torch_to_jax.register(abc.Mapping)
def _torch_dict_to_jax(
    value: Dict[str, Union[torch.Tensor, Any]]
) -> Dict[str, Union[DeviceArray, Any]]:
  """Converts a dict of PyTorch tensors into a dict of Jax DeviceArrays."""
  return type(value)(**{k: torch_to_jax(v) for k, v in value.items()})  # type: ignore


@functools.singledispatch
def jax_to_torch(value: Any, device: Device = None) -> Any:
  """Convert JAX values to PyTorch Tensors.
  By default, the returned tensors are on the same device as the Jax inputs,
  but if `device` is passed, the tensors will be moved to that device.
  """
  # Don't do anything by default, and when a handler is registered for this type
  # of value, it gets used to convert it to a torch tensor.
  # NOTE: The alternative would be to raise an error when an unsupported value
  # is encountered:
  # raise NotImplementedError(f"Cannot convert {v} to a Torch tensor")
  return value


@jax_to_torch.register(DeviceArray)
def _devicearray_to_tensor(value: DeviceArray,
                           device: Device = None) -> torch.Tensor:
  """Converts a Jax DeviceArray into PyTorch Tensor."""
  dpack = jax_dlpack.to_dlpack(value.astype("float32"))
  tensor = torch_dlpack.from_dlpack(dpack)
  if device:
    return tensor.to(device=device)
  return tensor


@jax_to_torch.register(abc.Mapping)
def _jax_dict_to_torch(
    value: Dict[str, Union[DeviceArray, Any]],
    device: Device = None) -> Dict[str, Union[torch.Tensor, Any]]:
  """Converts a dict of Jax DeviceArrays into a dict of PyTorch tensors."""
  return type(value)(
      **{k: jax_to_torch(v, device=device) for k, v in value.items()})  # type: ignore
  

class JaxToTorchWrapper(gym.Wrapper):
  """Wrapper that converts Jax tensors to PyTorch tensors."""

  def __init__(self,
               env: Union[wrappers.GymWrapper, wrappers.VectorGymWrapper],
               device: Optional[Device] = None):
    """Creates a Wrapper around a `GymWrapper` or `VectorGymWrapper` that outputs PyTorch tensors."""
    super().__init__(env)
    self.device: Optional[Device] = device

  def observation(self, observation):
    return jax_to_torch(observation, device=self.device)

  def action(self, action):
    return torch_to_jax(action)

  def reward(self, reward):
    return jax_to_torch(reward, device=self.device)

  def done(self, done):
    return jax_to_torch(done, device=self.device)

  def info(self, info):
    return jax_to_torch(info, device=self.device)

  def reset(self):
    obs = super().reset()
    return self.observation(obs)

  def step(self, action):
    action = self.action(action)
    obs, reward, done, info = super().step(action)
    obs = self.observation(obs)
    reward = self.reward(reward)
    done = self.done(done)
    info = self.info(info)
    return obs, reward, done, info