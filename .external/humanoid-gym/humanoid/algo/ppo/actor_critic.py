# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.utils import parametrizations, spectral_norm


class ScaledLinear(nn.Linear):
    def __init__(self, in_features, out_features, output_scale=1.0):
        super().__init__(in_features, out_features)
        self.output_scale = float(output_scale)

    def forward(self, input):
        output = super().forward(input)
        if self.output_scale == 1.0:
            return output
        return output * self.output_scale


class ActorCritic(nn.Module):
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        init_noise_std=1.0,
                        activation = nn.ELU(),
                        actor_spectral_norm=False,
                        actor_spectral_norm_output_layer=True,
                        actor_spectral_norm_layer_scope="all",
                        actor_spectral_norm_coeff=1.0,
                        actor_orthogonal_parametrization=False,
                        actor_orthogonal_output_layer=True,
                        actor_orthogonal_layer_scope="all",
                        actor_layer_norm=False,
                        actor_layer_norm_output_layer=False,
                        actor_layer_norm_layer_scope="hidden",
                        actor_output_gain=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()


        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        actor_spectral_norm = bool(actor_spectral_norm)
        actor_spectral_norm_output_layer = bool(actor_spectral_norm_output_layer)
        actor_spectral_norm_layer_scope = str(actor_spectral_norm_layer_scope).lower()
        actor_spectral_norm_coeff = float(actor_spectral_norm_coeff)
        actor_orthogonal_parametrization = bool(actor_orthogonal_parametrization)
        actor_orthogonal_output_layer = bool(actor_orthogonal_output_layer)
        actor_orthogonal_layer_scope = str(actor_orthogonal_layer_scope).lower()
        actor_layer_norm = bool(actor_layer_norm)
        actor_layer_norm_output_layer = bool(actor_layer_norm_output_layer)
        actor_layer_norm_layer_scope = str(actor_layer_norm_layer_scope).lower()
        actor_output_gain = float(actor_output_gain)
        if actor_spectral_norm_coeff <= 0:
            raise ValueError("actor_spectral_norm_coeff must be positive")
        if actor_output_gain <= 0:
            raise ValueError("actor_output_gain must be positive")
        valid_layer_scopes = {"all", "hidden", "first_hidden"}
        if actor_spectral_norm_layer_scope not in valid_layer_scopes:
            raise ValueError(f"actor_spectral_norm_layer_scope must be one of {sorted(valid_layer_scopes)}")
        if actor_orthogonal_layer_scope not in valid_layer_scopes:
            raise ValueError(f"actor_orthogonal_layer_scope must be one of {sorted(valid_layer_scopes)}")
        if actor_layer_norm_layer_scope not in valid_layer_scopes:
            raise ValueError(f"actor_layer_norm_layer_scope must be one of {sorted(valid_layer_scopes)}")
        if actor_spectral_norm and actor_orthogonal_parametrization:
            raise ValueError("actor_spectral_norm and actor_orthogonal_parametrization cannot both be enabled")

        def use_actor_spectral_norm(hidden_layer_index=None, is_output_layer=False):
            if not actor_spectral_norm:
                return False
            if is_output_layer:
                return actor_spectral_norm_output_layer and actor_spectral_norm_layer_scope == "all"
            if actor_spectral_norm_layer_scope in {"all", "hidden"}:
                return True
            return hidden_layer_index == 0

        def use_actor_orthogonal_parametrization(hidden_layer_index=None, is_output_layer=False):
            if not actor_orthogonal_parametrization:
                return False
            if is_output_layer:
                return actor_orthogonal_output_layer and actor_orthogonal_layer_scope == "all"
            if actor_orthogonal_layer_scope in {"all", "hidden"}:
                return True
            return hidden_layer_index == 0

        def use_actor_layer_norm(hidden_layer_index=None, is_output_layer=False):
            if not actor_layer_norm:
                return False
            if is_output_layer:
                return actor_layer_norm_output_layer and actor_layer_norm_layer_scope == "all"
            if actor_layer_norm_layer_scope in {"all", "hidden"}:
                return True
            return hidden_layer_index == 0

        # Policy
        actor_layers = []
        actor_layers.append(
            self._build_linear(
                mlp_input_dim_a,
                actor_hidden_dims[0],
                use_spectral_norm=use_actor_spectral_norm(hidden_layer_index=0),
                use_orthogonal_parametrization=use_actor_orthogonal_parametrization(hidden_layer_index=0),
                spectral_norm_coeff=actor_spectral_norm_coeff,
            )
        )
        if use_actor_layer_norm(hidden_layer_index=0):
            actor_layers.append(nn.LayerNorm(actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(
                    self._build_linear(
                        actor_hidden_dims[l],
                        num_actions,
                        use_spectral_norm=use_actor_spectral_norm(is_output_layer=True),
                        use_orthogonal_parametrization=use_actor_orthogonal_parametrization(is_output_layer=True),
                        spectral_norm_coeff=actor_spectral_norm_coeff,
                        output_scale=actor_output_gain,
                    )
                )
                if use_actor_layer_norm(is_output_layer=True):
                    actor_layers.append(nn.LayerNorm(num_actions))
            else:
                actor_layers.append(
                    self._build_linear(
                        actor_hidden_dims[l],
                        actor_hidden_dims[l + 1],
                        use_spectral_norm=use_actor_spectral_norm(hidden_layer_index=l + 1),
                        use_orthogonal_parametrization=use_actor_orthogonal_parametrization(hidden_layer_index=l + 1),
                        spectral_norm_coeff=actor_spectral_norm_coeff,
                    )
                )
                if use_actor_layer_norm(hidden_layer_index=l + 1):
                    actor_layers.append(nn.LayerNorm(actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        self.register_buffer("action_output_scale", torch.tensor(1.0))
        self.register_buffer("action_std_scale", torch.tensor(1.0))
        self.actor_spectral_norm = actor_spectral_norm
        self.actor_spectral_norm_output_layer = actor_spectral_norm_output_layer
        self.actor_spectral_norm_layer_scope = actor_spectral_norm_layer_scope
        self.actor_spectral_norm_coeff = actor_spectral_norm_coeff
        self.actor_orthogonal_parametrization = actor_orthogonal_parametrization
        self.actor_orthogonal_output_layer = actor_orthogonal_output_layer
        self.actor_orthogonal_layer_scope = actor_orthogonal_layer_scope
        self.actor_layer_norm = actor_layer_norm
        self.actor_layer_norm_output_layer = actor_layer_norm_output_layer
        self.actor_layer_norm_layer_scope = actor_layer_norm_layer_scope
        self.actor_output_gain = actor_output_gain
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
    @staticmethod
    def _build_linear(
        in_features,
        out_features,
        use_spectral_norm=False,
        use_orthogonal_parametrization=False,
        spectral_norm_coeff=1.0,
        output_scale=1.0,
    ):
        if use_spectral_norm:
            output_scale = spectral_norm_coeff
        layer = ScaledLinear(in_features, out_features, output_scale=output_scale)
        if use_spectral_norm:
            layer = spectral_norm(layer)
        if use_orthogonal_parametrization:
            layer = parametrizations.orthogonal(layer)
        return layer

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def set_action_output_scale(self, value):
        scale = float(value)
        if scale <= 0.0:
            raise ValueError("action_output_scale must be positive")
        self.action_output_scale.fill_(scale)

    def get_action_output_scale(self):
        return float(self.action_output_scale.item())

    def set_action_std_scale(self, value):
        scale = float(value)
        if scale <= 0.0:
            raise ValueError("action_std_scale must be positive")
        self.action_std_scale.fill_(scale)

    def get_action_std_scale(self):
        return float(self.action_std_scale.item())

    def set_policy_output_scale(self, value):
        self.set_action_output_scale(value)
        self.set_action_std_scale(value)

    def get_policy_output_scale(self):
        return float(self.action_output_scale.item())

    def update_distribution(self, observations):
        mean = self.actor(observations) * self.action_output_scale
        std = self.std * self.action_std_scale
        self.distribution = Normal(mean, mean*0. + std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations) * self.action_output_scale
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
