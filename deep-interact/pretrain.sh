cd ..;
schedulers=( ReduceLROnPlateau )
# schedulers=( ReduceLROnPlateau CosineAnnealingLR CosineAnnealingWarmRestarts )

for i in "${!schedulers[@]}";
do
    PYTHONPATH='.' python runners/pretrain_deep_interact.py --input_data_dir=/home/patrick/3d-2d-pretraining/data/molecule_datasets --dataset=QM9 --model_3d=SchNet --epochs 200 --output_model_dir ./deep-interact/assets --lr_scheduler ${schedulers[$i]} --batch_size 1 --verbose --device $((i)) --mode method --require_3d --num_workers 4 --process_num $i --output_model_name deep_interact
done

# wait

# python3 aggregate.py
# rm *_config.csv

