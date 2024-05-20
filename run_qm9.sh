pretraining_dataset=PCQM4Mv2
finetuning_dataset=QM9

blocks=8
bsz=128
model_2d=GIN

input_model_file=./deep-interact/assets/A_AB_ABC.pth # path to checkpoint

output_model_name=finetune_${finetuning_dataset}_from_${pretraining_dataset}_gin_schnet_L${blocks}_A_AB_ABC

PYTHONPATH='.' python3 runners/finetune_QM9_deep_interact.py --input_data_dir=data --dataset=QM9 --epochs 1000 \
 --input_model_file $input_model_file --output_model_dir ./deep-interact/assets  --lr_scheduler CosineAnnealingLR \
 --batch_size $bsz --verbose --device 0 --num_workers 8 --model_3d SchNet --model_2d $model_2d --gnn_type $model_2d \
 --num_interaction_blocks $blocks --output_model_name $output_model_name --dropout_ratio 0 --interaction_rep_2d mean \
 --interaction_rep_3d mean --interaction_agg cat --final_pool cat --mode method --emb_dim 300 --lr 1e-4 -\
 -initialization glorot_normal --input_model_file $input_model_file --save_config --verbose --diff_interactor_per_block \
  --task=gap \
  --residual --interactor_activation Swish \
 --interact_every_block \
