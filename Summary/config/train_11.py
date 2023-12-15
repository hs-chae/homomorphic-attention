# config for training 1 layer 1 head version.
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_11.py

wandb_log = False
wandb_project = '1 layer 1 head'
wandb_run_name= 'run_name' 

# these make the total batch size be ~0.5M
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 40

# this makes total number of tokens be 300B
max_iters = 10000
lr_decay_iters = max_iters
# eval stuff
eval_interval = 500
eval_iters = 100
log_interval = 10

# weight decay
weight_decay = 1e-1

compile=False
out_dir = 'out_11' #'config/GK-exp2poly-12lyrs-combined'

dataset = 'TinyStories' #'openwebtext'

bias=True
init_from = 'scratch' 
dropout = 0.0

attn_type="GK"
add_lora = False


#1 layer 1 head version
n_layer = 1
n_head = 1 #8  # 12   # <= TODO:  compare difference
n_embd = 768

