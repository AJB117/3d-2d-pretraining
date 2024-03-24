cd ..;
schedulers=( CosineAnnealingLR )
# schedulers=( ReduceLROnPlateau CosineAnnealingLR CosineAnnealingWarmRestarts )

if [[ $1 == "PCQM4Mv2" ]]; then
    input_data_dir=/home/patrick/data/
    dataset=PCQM4Mv2
else
    input_data_dir=data
    dataset=QM9
fi

for i in "${!schedulers[@]}";
do
    PYTHONPATH='.' python runners/pretrain_deep_interact.py --input_data_dir=$input_data_dir --dataset=$dataset --model_3d=SchNet --model_2d GIN --epochs 200 --output_model_dir ./deep-interact/assets --lr_scheduler ${schedulers[$i]} --batch_size 128 --verbose --device 2 --mode method --num_workers 6 --process_num $i --interaction_rep_2d mean --interaction_rep_3d mean --interaction_agg cat --final_pool cat --emb_dim 300 --layer_norm --residual --num_interaction_blocks 6 --output_model_name deep_interact_interatomic_edge_pred_20 --wandb
done
# edge_pred_# where # is the number of sampled negative links
# interatomic vs interatomic_# where # is the number of sampled distances

# wait

# python3 aggregate.py
# rm *_config.csv

