cd ..;
PYTHONPATH='.' python runners/pretrain.py --input_data_dir=/home/patrick/3d-2d-pretraining/data/molecule_datasets --dataset=QM9 --model_3d=EGNN --epochs 1 --output_model_dir ./egnn_qm9/assets --lr_scheduler ReduceLROnPlateau --batch_size 256 --verbose --device 1 --mode EGNN --require_3d --num_workers 4 --input_model_file_3d /home/patrick/3d-2d-pretraining/egnn_pcqm4mv2/assets/model.pth

