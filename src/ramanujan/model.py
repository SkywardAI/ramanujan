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


import torch
import inspect
import torch.nn as nn
from torch.nn import functional as F


######################
# Internal Libraries #
######################

from .attentions import CausalSelfAttention
from.configs import RanmanujanConfig
from .mlp import MLP


class RanmanujanBlock(nn.Module):
    """
    The transformer block of the Ranmanujan model
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embd)
        self.attn=CausalSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Ranmanujan(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd), # output embedding
                wpe=nn.Embedding(config.block_size, config.n_embd), # positional encoding
                h=nn.ModuleList([RanmanujanBlock(config) for _ in range(config.n_layer)]), # transformer layers => here is the masked multi-head attention(GPT2 version)
                ln_f=nn.LayerNorm(config.n_embd), # final layer normalization (linear layer)
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # fix the bug weight sharing scheme

        # model initialization: std=0.02, residual init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std=0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std*=(2*self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    # Implementing the forward pass to get logits
    def forward(self, idx, targets=None):
        # idx is of shape (B,T) batch dimension and time dimension
        B,T=idx.size()
        # T should less than the maximum sequence length
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # Generate a 1-dimensional tensor using torch.arange(start, end) to forward the token and position embeddings.
        pos=torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb=self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb=self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x=tok_emb+pos_emb # combine token and position embeddings
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x=block(x)
        # forward the final layernorm and the classifer
        x=self.transformer.ln_f(x)
        logits=self.lm_head(x) # shape (B, T, vocab_size)

        loss=None

        if targets is not None:
            loss=F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Load pretrained GPT-2 model weigths from HF"""
        assert model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']

        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained model: %s " % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        # the number of parameters
        config_args={
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), # 124M
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), # 774M
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600) # 1558M
        }[model_type]

        config_args['vocab_size']=50257 # always 50257 for GPT-2 model checkpoint
        config_args['block_size']=1024 # always 1024 for GPT-2 model checkpoint

        config=RanmanujanConfig(**config_args)
        model=Ranmanujan(config)
        sd=model.state_dict()
        sd_keys=sd.keys()
        sd_keys=[k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask /buffer, not a parameter

        # init a HF model
        model_hf=GPT2LMHeadModel.from_pretrained(model_type)

        # state_dict is a dictionary of the model's parameters, which can be loaded into the model using its load_state_dict method
        sd_hf=model_hf.state_dict()

        # copy while ensurigng all of the parameters are aligned and match in names and shapes
        sd_keys_hf=sd_hf.keys()

        sd_keys_hf=[k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf=[k for k in sd_keys_hf if not k.endswith('attn.bias')] # same, just the mask (buffer)
        transposed=['attn.c_attn.weight', 'attn.c_proj.weight','mlp.c_fc.weight','mlp.c_proj.weight'] # hardcode layers from tensorflow to pytorch

        # basically the openai checlpoints use a "Conv1D" module, but we only want to use a vanilla Linear this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf)==len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1]==sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape==sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    

    # add weight decaym only for 2D params, and add fused AdamW
    def configure_optimizers(self, weight_decay, learning_rate, device, master_process=True):
        # start with all of the candidate parameters
        param_dict={pn: p for pn, p in self.named_parameters()}
        param_dict={pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups.
        # Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e all weight tensors in matmul + embeddings decay, all biases and layernorms do not
        decay_params=[p for n,p in param_dict.items() if p.dim() >= 2 ]
        nodecay_params=[p for n,p in param_dict.items() if p.dim() < 2 ]
        optim_groups=[
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params=sum(p.numel() for p in decay_params)
        num_nodecay_params=sum(p.numel() for p in nodecay_params)
        
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Create AdamW optimizer and use the fused version if it is available
        fused_avaliable='fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused=fused_avaliable and 'cuda' in device

        if master_process:
            print(f"using fused AdamW: {use_fused}")

        optimizer=torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        return optimizer
