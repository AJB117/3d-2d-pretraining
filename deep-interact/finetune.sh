cd ..;
schedulers=( ReduceLROnPlateau CosineAnnealingLR StepLR CosineAnnealingWarmRestarts )
targets=( mu alpha homo lumo gap r2 zpve u0 u298 h298 g298 cv gap_02 )

# for i in "${!schedulers[@]}";
# do
PYTHONPATH='.' python runners/finetune_QM9_deep_interact.py --task=gap --input_data_dir=/home/patrick/3d-2d-pretraining/data/molecule_datasets --dataset=QM9 --epochs 1000 --output_model_dir ./deep-interact/assets  --lr_scheduler CosineAnnealingLR --batch_size 128 --verbose --device 0 --num_workers 8 --mode method --model_3d SchNet --model_2d GIN --num_interaction_blocks 6 --output_model_name deep_interact_schnet_gin_6_layers
# done

# method
# PYTHONPATH='.' python runners/finetune_QM9.py --task=gap --input_data_dir=/home/patrick/3d-2d-pretraining/data/molecule_datasets --dataset=QM9 --use_2d --epochs 1 --output_model_dir ./method/assets  --input_model_file ./method/assets/model_complete_use.pth --lr_scheduler CosineAnnealingLR --batch_size 128 --verbose --device 1 --num_workers 4 --mode method --use_3d --model_3d EGNN

# baseline egnn
# PYTHONPATH='.' python runners/finetune_QM9.py --task=gap --input_data_dir=/home/patrick/3d-2d-pretraining/data/molecule_datasets --dataset=QM9 --use_2d --epochs 1 --output_model_dir ./method/assets  --lr_scheduler CosineAnnealingLR --batch_size 128 --verbose --device 1 --num_workers 4 --mode baseline --use_3d --model_3d EGNN
