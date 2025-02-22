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
import math
import time
import torch
import torch.distributed as dist

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP 


from ramanujan.data_loader import DataLoaderLite
from ramanujan.model import Ranmanujan
from ramanujan.configs import RanmanujanConfig
from utils.file_loader import load_data



os.environ['RANK']='-1'
os.environ['LOCAL_RANK']='0' # multiple nodes with multiple GPUs
os.environ['WORLD_SIZE']='2' # the number of working process on GPU, how many of GPUs
# os.environ['MASTER_ADDR']='localhost'
# os.environ['MASTER_PORT']='12355'

# For tracking fake tensor usage issue
os.environ["TORCH_FAKE_TENSOR_DEBUG"] = "1"

ddp=int(os.environ.get('RANK','-1')) != -1

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

if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


current_dir=os.path.dirname(os.path.abspath(__file__))
file_path=os.path.join(current_dir,'..','data','input.txt')

train_loader = DataLoaderLite(
    B=B,
    T=T,
    text=load_data(file_path),
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    master_process=master_process
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


# define learning rate schedule
max_lr=6e-4
min_lr=max_lr*0.1
warmup_steps=10
max_steps=50

# add learning rate sheduler
def get_lr(it):
    # linear warmup for warmup_iters steps
    if it<warmup_steps:
        return max_lr*(it+1)/warmup_steps
    # if it>lr_decay_iters, return min learning rate
    if it>max_steps:
        return min_lr
    # in between, use cosine decay down to min learning rate
    decay_ratio=(it-warmup_steps)/(max_steps-warmup_steps)
    assert 0 <=decay_ratio<=1
    coeff=0.5*(1.0+math.cos(math.pi*decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr+coeff*(max_lr-min_lr)

# optimize
optimizer=raw_model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=6e-4,
    device=device,
)

# training loop
for step in range(max_steps):
    t0=time.time()
    optimizer.zero_grad()

    loss_accum=0.0
    for micro_step in range(grad_accum_steps):
        x,y=train_loader.next_batch()
        x,y=x.to(device),y.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss=model(x,y)
        
        loss=loss/grad_accum_steps
        loss_accum+=loss.detach()
        if ddp:
            model.require_backward_grad_sync=(micro_step==grad_accum_steps-1)
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm=torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determine and set the learning rate for this iteration
    lr=get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr']=lr
    
    optimizer.step()

    torch.cuda.synchronize()
    t1=time.time()

    dt=t1-t0 # time difference in seconds
    tokens_processed=train_loader.B*train_loader.T*grad_accum_steps*ddp_world_size
    tokens_per_sec=tokens_processed/dt

    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()
