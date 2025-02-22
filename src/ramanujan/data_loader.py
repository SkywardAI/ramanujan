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
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T, text, process_rank, num_processes):
        self.B=B
        self.T=T
        self.process_rank=process_rank
        self.num_processes=num_processes

        # At the init load tokens from disj and store them in memory
        enc=tiktoken.get_encoding(text)
        tokens=enc.encode(text)
        self.tokens=torch.tensor(tokens)

        self.current_position=self.B*self.T*self.process_rank
    
    def next_batch(self):
        B,T=self.B,self.T
        buf=self.tokens[self.current_position: self.current_position+B*T+1]
        x=(buf[:-1].view(B,T)) # inputs
        y=(buf[1:].view(B,T)) # targets

        # advance the position in the tensor
        self.current_position+=B*T*self.num_processes

        # if loading the next batch would be out of bounds, reset the position
        if self.current_position+(B*T*self.num_processes+1)>len(self.tokens):
            self.current_position=self.B*self.T*self.process_rank
        return x,y
