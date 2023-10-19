#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from core import CRLAgents

from .brax_agents import *
from .packnet_agents import *
from .pnn_agents import *
from .single_agents import *
from .subspace_agents import *
from .rewire_agents import *


def ActionAgent(input_dimension,output_dimension, hidden_size, start_steps, layer_norm):
    """OneStep model: a single policy that is re-used and fine-tuned over the task sequences.
    """
    return CRLAgents(Action(input_dimension,output_dimension, hidden_size, start_steps, input_name = "env/env_obs", layer_norm = layer_norm))

def MultiActionAgent(input_dimension,output_dimension, hidden_size, start_steps, layer_norm):
    """Fine-tune and clone: the model is saved when the task is ended, and duplicated to be fine-tuned on the next task.
    """
    return CRLAgents(MultiAction(input_dimension,output_dimension, hidden_size, start_steps, input_name = "env/env_obs", layer_norm = layer_norm))

def FromScratchActionAgent(input_dimension,output_dimension, hidden_size, start_steps, layer_norm):
    """From scratch: the model is saved when the task is ended, and a new random one is created for the next task.
    """
    return CRLAgents(FromScratchAction(input_dimension,output_dimension, hidden_size, start_steps, input_name = "env/env_obs", layer_norm = layer_norm))

def EWCActionAgent(input_dimension,output_dimension, hidden_size, fisher_coeff, start_steps, layer_norm):
    """EWC regularizer added in the framework.
    """
    return CRLAgents(EWCAction(input_dimension,output_dimension, hidden_size, fisher_coeff, start_steps, input_name = "env/env_obs", layer_norm = layer_norm))

def L2ActionAgent(input_dimension,output_dimension, hidden_size, l2_coeff, start_steps, layer_norm):
    """L2 regularizer added in the framework.
    """
    return CRLAgents(L2Action(input_dimension,output_dimension, hidden_size, l2_coeff, start_steps, input_name = "env/env_obs", layer_norm = layer_norm))

def PNNActionAgent(input_dimension,output_dimension, hidden_size, start_steps, layer_norm):
    """PNN Agent 
    """
    return CRLAgents(PNNAction(input_dimension,output_dimension, hidden_size, start_steps, input_name = "env/env_obs"))

def PacknetActionAgent(input_dimension,output_dimension, hidden_size, start_steps, layer_norm, prune_coeff):
    """Packnet Agent 
    """
    return CRLAgents(PacknetAction(input_dimension,output_dimension, hidden_size, start_steps, input_name = "env/env_obs", layer_norm = layer_norm, prune_coeff = prune_coeff))

def TwinCritics(obs_dimension, action_dimension, hidden_size, layer_norm):
    """Twin q value functions for SAC algorithm.
    """
    return CRLAgents(Critic(obs_dimension, action_dimension, hidden_size, input_name = "env/env_obs",output_name = "q1", layer_norm = layer_norm),
                     Critic(obs_dimension, action_dimension, hidden_size, input_name = "env/env_obs", output_name = "q2", layer_norm = layer_norm))

def SubspaceActionAgent(n_initial_anchors, dist_type, refresh_rate, input_dimension,output_dimension, hidden_size, start_steps, resampling_q, resampling_policy, repeat_alpha, layer_norm):
    """ActionAgent that is using "alphas" variable during forward to compute a convex combination of its anchor policies.
    """
    return SubspaceAgents(AlphaAgent(n_initial_anchors, dist_type, refresh_rate, resampling_q, resampling_policy, repeat_alpha),
                          SubspaceAction(n_initial_anchors,input_dimension,output_dimension, hidden_size, start_steps))

def AlphaTwinCritics(n_anchors, obs_dimension, action_dimension, hidden_size, layer_norm):
    """Twin critics model used for SAC. In addition to the (obs,actions), they also take the convex combination alpha as as input.
    """
    return SubspaceAgents(AlphaCritic(n_anchors, obs_dimension, action_dimension, hidden_size, output_name = "q1"),
                          AlphaCritic(n_anchors, obs_dimension, action_dimension, hidden_size, output_name = "q2"))

def RewireActionAgent(input_dimension,output_dimension, hidden_size, start_steps, layer_norm, tau, beta, k, tau2, beta2, rewire_last, cycle, divergence, coeff):
    """ActionAgent that is using "alphas" variable during forward to compute a convex combination of its anchor policies.
    """
    return CRLAgents(RewireAction(input_dimension,output_dimension, hidden_size, start_steps, tau=tau, beta=beta, k=k, tau2=tau2, beta2=beta2, rewire_last=rewire_last, cycle=cycle, divergence=divergence, coeff=coeff))
