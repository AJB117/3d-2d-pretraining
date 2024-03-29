import argparse
import os


def main_cmd(args, body, task, dir_name):
    if args.machine == "rivanna":
        home_dir = "/home/zqe3cg/3d-2d-pretraining"
        source_str = "source $SCRATCH_DIR/.virtualenvs/3d-pretraining/bin/activate"
        data_path = "data"
    elif args.machine == "portal01":
        home_dir = "/u/zqe3cg/3d-2d-pretraining"
        source_str = "source /p/3dpretraining/molecules/bin/activate"
        data_path="/p/3dpretraining/3d-pretraining/data"

    body += f"""
#SBATCH --output=\"{home_dir}/{dir_name}/logfiles/{args.mode}_{args.dataset}_{args.job_name}_{task}.log\""""
    body += f"""
#SBATCH --error=\"{home_dir}/{dir_name}/logfiles/{args.mode}_{args.dataset}_{args.job_name}_{task}.err\""""
    body += f"""
#SBATCH --job-name="{args.mode}_{args.dataset}_{args.job_name}_{task}\""""
    body += f"""
echo "Hostname -> $HOSTNAME"

source /etc/profile.d/modules.sh

{source_str}

echo "which python -> $(which python)"
nvidia-smi

echo "STARTIME $i $(date)"
    """

    body += f"""
cd ../../;

PYTHONPATH='.' python3 runners/finetune_QM9_deep_interact.py --task={task} --input_data_dir={data_path} --dataset={args.dataset} --epochs 1000 --output_model_dir ./deep-interact/assets  --lr_scheduler CosineAnnealingLR --batch_size 128 --verbose --num_workers {args.num_workers} --mode method --model_3d SchNet --model_2d GIN --num_interaction_blocks {args.num_blocks} --output_model_name {args.output_model_name}_{task} --interaction_rep_2d mean --interaction_rep_3d mean --interaction_agg cat --final_pool cat --mode method --emb_dim 300 --lr 1e-4 --layer_norm --residual --initialization glorot_normal --input_model_file ./deep-interact/assets/{args.input_model_file} --no_verbose {'--wandb' if args.wandb else ""}"""

    body += """
echo "ENDTIME $i $(date)"
"""

    return body


def main(args):
    tasks = [
        "mu",
        "alpha",
        "homo",
        "lumo",
        "r2",
        "zpve",
        "u0",
        "u298",
        "h298",
        "g298",
        "cv",
        "gap_02",
    ]

    if args.machine == "rivanna":
        header = f"""#!/bin/bash -l

# --- Resource related ---
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=30000
#SBATCH -c {args.num_workers}

# --- Task related ---
    """
    elif args.machine == "portal01":
        header = f"""#!/bin/bash -l

# --- Resource related ---
#SBATCH --partition="gpu"
#SBATCH --gpus-per-node=1
#SBATCH --requeue                   ### On failure, requeue for another try
#SBATCH --exclude=adriatic04,lynx11,cheetah02,adriatic06,affogato11,lynx04,lynx05,affogato15,adriatic01,lynx10,adriatic03,ristretto01,adriatic02,affogato13,jaguar05,affogato12,lynx02,lynx03,lynx12,lynx06,affogato14,lynx07,lynx01,adriatic05,cheetah01
"""

    dir_name = f"deep-interact/{args.output_model_name}_slurm"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    logfiles_dir = f"{dir_name}/logfiles"
    if not os.path.exists(logfiles_dir):
        os.makedirs(logfiles_dir)

    for task in tasks:
        main_body = main_cmd(args, "", task, dir_name)
        with open(f"{dir_name}/{args.output_model_name}_{task}.sh", "w+") as f:
            f.write(header)
            f.write(main_body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_model_file")
    parser.add_argument("--output_model_name")
    parser.add_argument("--num_workers", default=6)
    parser.add_argument("--job_name")
    parser.add_argument("--dataset", default="QM9")
    parser.add_argument("--mode", default="tune")
    parser.add_argument("--num_blocks")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--machine", default="rivanna")

    args = parser.parse_args()
    main(args)
