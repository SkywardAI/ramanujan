# coding=utf-8

# Copyright [2025] [SkywardAI]
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # it projects the input from the original dimensionality to four times that size which is c_fc
        self.c_fc=nn.Linear(config.n_embd, 4*config.n_embd)
        # GELU activation function with tanh approximation
        # tanh approximation is more efficient than the original GELU
        self.gelu=nn.GELU(approximate='tanh')
        # it projects the input back to the original dimensionality
        self.c_proj=nn.Linear(4*config.n_embd, config.n_embd)
        # add NANOGPT_SCALE_INIT=1 to c_proj layer for initialization of scale parameter
        self.c_proj.NANOGPT_SCALE_INIT=1
    
    def forward(self, x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x


