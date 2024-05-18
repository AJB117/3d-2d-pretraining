# datasets=( bbbp clintox hiv muv tox21 toxcast sider bace )
dataset=tox21
bsz=256
lr=3e-4


input_model_file=./deep-interact/assets/
# input_model_file=./deep-interact/assets/spd_as_2-nodiscount.pth
output_model_name=tuning_test${dataset}
# for i in "${!datasets[@]}";
# do
PYTHONPATH='.' python3 runners/finetune_MoleculeNet_deep_interact.py --input_data_dir=data/MoleculeNet --dataset=$dataset --epochs 100 --output_model_dir ./deep-interact/assets  --lr_scheduler CosineAnnealingLR --batch_size $bsz --verbose --device 0 --num_workers 6 --model_3d SchNet --model_2d GIN --num_interaction_blocks 8 --output_model_name ${output_model_name}_${datasets[i]} --interaction_rep_2d mean --interaction_rep_3d mean --interaction_agg cat --final_pool cat --emb_dim 300 \
  --lr $lr --residual --initialization glorot_normal --input_model_file $input_model_file \
  --mode method --use_2d_only --num_layer 8 --gnn_type GIN --diff_interactor_per_block \
  --seed 42 --transfer --dropout_ratio 0.0 --JK mean
# done
