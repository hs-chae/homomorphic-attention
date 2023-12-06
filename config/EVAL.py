# evaluate the base gpt2
# n_layer=12, n_head=12, n_embd=768
# 124M parameters
batch_size = 8
eval_iters = 300 # use more iterations to get good estimate
eval_only = True
wandb_log = True





dataset = 'openwebtext' 
# dataset = 'lambada' 
# dataset = 'wiki' 
# dataset = 'ptb' 
# dataset = 'TinyStories' 


init_from = 'finetune'  #'gpt-2' #'manipulate'




wandb_run_name= 'prbllora' + dataset  #'gpt2-proj-block-12hd-weighted'#'only_qk_gpt2_MSE_combined' #'CRAMMED-GK-exp-12lyr-12-40-ddp'
wandb_project = 'finetune_score' #'FineTune Investigation' #'AttentionTransfer3'
out_dir = "LORATEST"#'TheFuture/small-prbl-lora' #'LeResult/small-full_ft-4days'

attn_type= "GK"
add_lora = True