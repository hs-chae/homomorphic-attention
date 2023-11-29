# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'layerwise distillation test'#'FineTune Investigation' #'AttentionTransfer3'
wandb_run_name= 'small-prbl-weighted' #'gpt2-proj-block-12hd-weighted'#'only_qk_gpt2_MSE_combined' #'CRAMMED-GK-exp-12lyr-12-40-ddp'

# these make the total batch size be ~0.5M
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 40

# this makes total number of tokens be 300B
max_iters = 100000
lr_decay_iters = 100000
# eval stuff
eval_interval = 25
eval_iters = 2000
log_interval = 10

# weight decay
weight_decay = 1e-1

compile=False
out_dir = 'TheFuture/small-prbl-weighted' #'config/GK-exp2poly-12lyrs-combined'

dataset = 'openwebtext'

bias=True
init_from = 'manipulate' #'finetune_engineer' #'engineer'
dropout = 0.0



# #gpt-2 base
n_layer = 12
n_head = 12
n_embd=768

# 'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
#             'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
#             'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
#             'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params

# n_layer=24
# n_head=16
# n_embd=1024

# n_layer=36
# n_head=20
# n_embd=1280

# n_layer=48
# n_head=25
# n_embd=1600


# init_from = 'scratch'
# wandb_project = 'ft_TinyStories'
# wandb_run_name= '1block_2to6_myGELU2' #'gpt2-proj-block-12hd-weighted'#'only_qk_gpt2_MSE_combined' #'CRAMMED-GK-exp-12lyr-12-40-ddp
# n_layer = 1
# n_head = 1
# n_embd=768
# out_dir = 'attn_transfer/1block-2to6-myGELU2' #'config/GK-exp2poly-12lyrs-combined
# dataset= 'TinyStories'
