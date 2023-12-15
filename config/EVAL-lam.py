# evaluate the base gpt2
# n_layer=12, n_head=12, n_embd=768
# 124M parameters
batch_size = 8
eval_iters = 300 # use more iterations to get good estimate
eval_only = True
wandb_log = True





dataset = 'openwebtext' 
# dataset = 'wiki' 
dataset = 'lambada' 
# dataset = 'ptb' 
# dataset = 'TinyStories' 


# init_from = 'resume'
init_from = 'finetune' #'finetune'  #'gpt-2' #'manipulate'





wandb_project = 'finetune_score' #'FineTune Investigation' #'AttentionTransfer3'
out_dir ='TheFuture/extreme-small-prbl-21k'# 'TheFuture/prep-small-prbl-2k-700'#'TheFuture/small-prbl-3350-500' #'TheFuture/prep-small-prbl-2k-300' #'LeResult/small-full_ft-6dyas'  #"TheFuture/small-prbl-PA-relu_square" #"LORATEST"#'TheFuture/small-prbl-lora' #'LeResult/small-full_ft-4days'
wandb_run_name= out_dir + '-' + dataset  #'gpt2-proj-block-12hd-weighted'#'only_qk_gpt2_MSE_combined' #'CRAMMED-GK-exp-12lyr-12-40-ddp'

attn_type= "GK"
add_lora = False


