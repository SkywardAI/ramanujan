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


import os
import torch
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP 


from .data_loader import DataLoaderLite
from .model import Ranmanujan
from .configs import RanmanujanConfig



os.environ['RANK']='-1'
os.environ['LOCAL_RANK']='0' # multiple nodes with multiple GPUs
os.environ['WORLD_SIZE']='2' # the number of working process on GPU, how many of GPUs
# os.environ['MASTER_ADDR']='localhost'
# os.environ['MASTER_PORT']='12355'


ddp=int(os.environ('RANK','-1')) != -1

if ddp:
    # use if DDP atm demands CUDA, we set the device appropriately accoridng to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank=int(os.environ['RANK'])
    ddp_local_rank=int(os.environ['LOCAL_RANK'])
    ddp_world_size=int(os.environ['WORLD_SIZE'])
    device=f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process=ddp_rank==0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank=0
    ddp_local_rank=0
    ddp_world_size=2
    master_process=True
    # attempt to autodetect device
    device="cpu"
    if torch.cuda.is_available():
        device="cuda"
    elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
        device="mps"
    print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Accoridng to ChatGPT3 paper, the batch_size of small model is 0.5M above 2**19 in number of tokens
total_batch_size=524288 # it is too large to use so we import gradient accumulation
B=4 # micro batch size
T=1024 # sequence length

# assert total_batch_size % (B*T)==0, "total batch size must be divisible by B*T"

assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"

grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

train_loader = DataLoaderLite(
    B=B,
    T=T,
    text="data/lotr.txt",
    process_rank=ddp_rank,
    num_processes=ddp_world_size
)

torch.set_float32_matmul_precision('medium')

model=Ranmanujan(
    config=RanmanujanConfig(
        vocab_size=50304
    )
)

model.eval()
model.to(device)
model=torch.compile(model)

if ddp:
    model=DDP(model, device_ids=[ddp_local_rank])

raw_model=model.module if ddp else model