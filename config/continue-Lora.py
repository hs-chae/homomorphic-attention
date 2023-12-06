import time


eval_interval = 5
eval_iters = 100
wandb_log = True # feel free to turn on
wandb_project = '2nd-tuning'#'Finetune-long'
dataset = 'openwebtext'
init_from =  'continued_gpt2' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 12
gradient_accumulation_steps = 32
max_iters = 1000

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False

# dtype = 'float32'

wandb_run_name = f'small-prbl-lora-3000'
out_dir = 'LeResult/small-prbl-lora'#'TheFuture/small-lora_qkv-3000'#'LORATEST'  # 'config/Approx12-ft32-12lyr-10k-TinyStories'
attn_type = 'GK'#'GKpoly-10'
add_lora = True