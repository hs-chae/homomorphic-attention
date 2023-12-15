# evaluate the base gpt2
# n_layer=12, n_head=12, n_embd=768
# 124M parameters
batch_size = 8
eval_iters = 500 # use more iterations to get good estimate
eval_only = True
wandb_log = False





# dataset = 'openwebtext' 
# dataset = 'lambada' 
# dataset = 'wiki' 
# dataset = 'ptb' 
dataset = 'TinyStories' 'gpt2'


init_from = 'resume'  #'gpt-2' #'manipulate'




wandb_run_name= 'small-prbl-weighted' + dataset 
wandb_project = 'finetune_score' 
out_dir = 'attn_transfer/second-12hd-2nd'