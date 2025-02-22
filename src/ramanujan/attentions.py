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


import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        # number of embedding dimensions must be divisible by number of heads
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # add NANOGPT_SCALE_INIT=1 to c_proj layer for initialization of scale parameter
        # @Bowen: Explain more
        self.c_proj.NANOGPT_SCALE_INIT=1
        # regularization
        self.n_head = config.n_head
        self.n_embd=config.n_embd
        # not really a bias more of a mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size,)).view(1, 1, config.block_size, config.block_size)
        )
    
    def forward(self, x):
        B,T,C=x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, value for all heads in batch and move head forward to be the batch dimension
        # nh is "number of heads"
        # hs is "head size"
        # C is number of channels = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer

        qkv=self.c_attn(x)
        q,k,v=qkv.split(self.n_embd, dim=2)
        k=k.view(B,T,self.n_head, C//self.n_head).transpose(1,2) # (B,nh,T,hs)
        q=q.view(B,T,self.n_head, C//self.n_head).transpose(1,2)
        v=v.view(B,T,self.n_head, C//self.n_head).transpose(1,2)

        # attention (materializes the large (T,T) matrix for all the queries and keys)
        # att=(q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        # att=att.masked_fill(self.bias[:,:,:T,:T] ==0, float('-inf'))
        # att=F.softmax(att, dim=-1)
        # y=att @ v # (B,nh,T,T)x(B,nh,T,hs)->(B,nh,T,hs)
        
        y=F.scaled_dot_product_attention(q,k,v, is_causal=True) # switch to flash attention
        ###########################################################################

        y=y.transpose(1,2).contiguous().view(B,T,C) # re-assemble all head outputs side by side
        # output projection
        y=self.c_proj(y)
        return y


    
