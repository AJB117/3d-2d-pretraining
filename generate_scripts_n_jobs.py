import pdb
import yaml
import argparse
import os
from typing import List


def n_tuples(lst, n):
    return [tuple(lst[i : i + n]) for i in range(0, len(lst), n)]


def redirect_str(err_fname, out_fname):
    if os.path.exists(err_fname):
        os.remove(err_fname)
    if os.path.exists(out_fname):
        os.remove(out_fname)

    return f""" 1> {out_fname} 2> {err_fname}"""


def main_body(dataset, sweep_dir, config_name):
    if dataset == "QM9":
        return f"""PYTHONPATH="." CUDA_LAUNCH_BLOCKING=1 python3 runners/finetune_QM9_deep_interact.py --config_dir {sweep_dir} --config_name {config_name}"""
    elif dataset in ("bace" "sider" "muv" "tox21" "hiv" "toxcast" "bbbp" "clintox"):
        return f"""PYTHONPATH="." CUDA_LAUNCH_BLOCKING=1 python3 runners/finetune_MoleculeNet_deep_interact.py --config_dir {sweep_dir} --config_name {config_name}"""
    elif dataset == "PCQM4Mv2":
        return f"""PYTHONPATH="." CUDA_LAUNCH_BLOCKING=1 python3 runners/finetune_PCQM4Mv2_deep_interact.py --config_dir {sweep_dir} --config_name {config_name}"""
    elif dataset == "Molecule3D":
        return f"""PYTHONPATH="." CUDA_LAUNCH_BLOCKING=1 python3 runners/finetune_Molecule3D_deep_interact.py --config_dir {sweep_dir} --config_name {config_name}"""
    else:
        raise ValueError(f"Dataset {dataset} not supported")


def main_cmd(config_names, job_name, body, hparams, sweep_dir, machine, dataset, mode):
    sweep_dir = sweep_dir

    if machine == "rivanna":
        home_dir = "/home/zqe3cg/3d-2d-pretraining"
        source_str = "source $SCRATCH_DIR/.virtualenvs/3d-pretraining/bin/activate"
    elif machine == "portal":
        home_dir = "/u/zqe3cg/3d-2d-pretraining"
        source_str = "source /p/3dpretraining/molecules/bin/activate"

    body += f"""
#SBATCH --output=\"{home_dir}/{sweep_dir}/logfiles/{mode}_{dataset}_{job_name}_{hparams}.log\""""
    body += f"""
#SBATCH --error=\"{home_dir}/{sweep_dir}/logfiles/{mode}_{dataset}_{job_name}_{hparams}.err\""""
    body += f"""
#SBATCH --job-name="{mode}_{dataset}_{job_name}_{hparams}\""""
    body += f"""
echo "Hostname -> $HOSTNAME"

source /etc/profile.d/modules.sh

{source_str}

echo "which python -> $(which python)"
nvidia-smi

echo "STARTIME $i $(date)"

cd ../../../;
"""

    hparam_arr = hparams.split("_")
    for i, config_name in enumerate(config_names):
        body += main_body(dataset, sweep_dir, config_name)
        body += redirect_str(
            f"{home_dir}/{sweep_dir}/logfiles/{mode}_{dataset}_{job_name}_{hparam_arr[i]}_custom.err",
            f"{home_dir}/{sweep_dir}/logfiles/{mode}_{dataset}_{job_name}_{hparam_arr[i]}_custom.log",
        )
        if i != len(config_names) - 1:
            body += "\\\n & \\\n"

    body += "\nwait;"
    body += '\necho "ENDTIME $i $(date)"'

    return body


def main(args):
    machine = os.popen("hostname").read()
    if machine.startswith("udc"):
        machine = "rivanna"
    elif machine.startswith("portal"):
        machine = "portal"

    if machine == "rivanna":
        header = """#!/bin/bash -l

# --- Resource related ---
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=50000
#SBATCH -c 24

# --- Task related ---
    """
    elif machine == "portal":
        header = """#!/bin/bash -l

# --- Resource related ---
#SBATCH --partition="gpu"
#SBATCH --gpus-per-node=1
#SBATCH --requeue                   ### On failure, requeue for another try
#SBATCH --exclude=adriatic04,lynx11,cheetah02,adriatic06,affogato11,lynx04,lynx05,affogato15,adriatic01,lynx10,adriatic03,ristretto01,adriatic02,affogato13,jaguar05,affogato12,lynx02,lynx03,lynx12,lynx06,affogato14,lynx07,lynx01,adriatic05,cheetah01
"""

    logfiles_dir = f"{args.sweep_dir}/logfiles"
    if not os.path.exists(logfiles_dir):
        os.makedirs(logfiles_dir)

    if not args.job_name:
        args.job_name = args.sweep_dir.split("/")[-1]

    sweep_configs: List[str] = []

    for sweep_config in os.listdir(args.sweep_dir):
        if not sweep_config.endswith(".yml"):
            continue
        sweep_configs.append(sweep_config)

    n = args.jobs_per_gpu
    mode = None
    dataset = None

    dir_for_tuples = os.path.join(args.sweep_dir, f"multi_job_scripts_{n}")
    if not os.path.exists(dir_for_tuples):
        os.makedirs(dir_for_tuples)

    for config_tuple in n_tuples(sweep_configs, n):
        hparams = []
        for config_name in config_tuple:
            with open(os.path.join(args.sweep_dir, config_name), "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                if "config_name" not in config:
                    raise ValueError("config_name not in config file")

                if mode is not None and config["mode"] != mode:
                    raise ValueError("not all jobs have the same mode")
                if dataset is not None and config["dataset"] != dataset:
                    raise ValueError("not all jobs have the same dataset")

                mode = config["mode"]
                dataset = config["dataset"]

                hparam = config_name.split("_")[-1].split(".yml")[0]
                hparams.append(hparam)

        hparam_str = "_".join(hparams)
        print(hparam_str)

        body = ""
        body = main_cmd(
            config_tuple,
            args.job_name,
            body,
            hparam_str,
            args.sweep_dir,
            machine,
            dataset,
            mode,
        )

        with open(f"{dir_for_tuples}/{args.job_name}_{hparam_str}.sh", "w+") as f:
            f.write(header)
            f.write(body)

    print(f"Saved all multi-job scripts to {dir_for_tuples}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--jobs_per_gpu",
        type=int,
        help="# of jobs per GPU to run concurrently",
    )
    parser.add_argument("--base_dir", type=str, default="./deep-interact")
    parser.add_argument("--sweep_dir", type=str)
    parser.add_argument("--job_name", type=str)

    args = parser.parse_args()
    main(args)
