import yaml
import argparse
import os


def main_cmd(config, job_name, body, hparam, sweep_dir, machine):
    config_name = config["config_name"]
    sweep_dir = sweep_dir

    if machine == "rivanna":
        home_dir = "/home/zqe3cg/3d-2d-pretraining"
        source_str = "source $SCRATCH_DIR/.virtualenvs/3d-pretraining/bin/activate"
    elif machine == "portal":
        home_dir = "/u/zqe3cg/3d-2d-pretraining"
        source_str = "source /p/3dpretraining/molecules/bin/activate"

    body += f"""
#SBATCH --output=\"{home_dir}/{sweep_dir}/logfiles/{config["mode"]}_{config["dataset"]}_{job_name}_{hparam}.log\""""
    body += f"""
#SBATCH --error=\"{home_dir}/{sweep_dir}/logfiles/{config["mode"]}_{config["dataset"]}_{job_name}_{hparam}.err\""""
    body += f"""
#SBATCH --job-name="{config["mode"]}_{config["dataset"]}_{job_name}_{hparam}\""""
    body += f"""
echo "Hostname -> $HOSTNAME"

source /etc/profile.d/modules.sh

{source_str}

echo "which python -> $(which python)"
nvidia-smi

echo "STARTIME $i $(date)"
    """

    if config["dataset"] == "QM9":
        body += f"""
cd ../../;
PYTHONPATH='.' python3 runners/finetune_QM9_deep_interact.py --config_dir {sweep_dir} --config_name {config_name}"""
    elif config["dataset"] in (
        "bace" "sider" "muv" "tox21" "hiv" "toxcast" "bbbp" "clintox"
    ):
        body += f"""
cd ../../;
PYTHONPATH='.' python3 runners/finetune_MoleculeNet_deep_interact.py --config_dir {sweep_dir} --config_name {config_name}"""

    body += "\necho 'ENDTIME $i $(date)'"

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
#SBATCH --mem-per-cpu=30000
#SBATCH -c 8

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

    for sweep_config in os.listdir(args.sweep_dir):
        if sweep_config.endswith(".yml"):
            with open(os.path.join(args.sweep_dir, sweep_config), "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                if "config_name" not in config:
                    raise ValueError("config_name not in config file")

                config_name = config["config_name"]
                hparam = config_name.split("_")[-1].split(".yml")[0]

                body = ""
                body = main_cmd(
                    config, args.job_name, body, hparam, args.sweep_dir, machine
                )
                with open(f"{args.sweep_dir}/{config_name}_{hparam}.sh", "w+") as f:
                    f.write(header)
                    f.write(body)

    print(f"Saved all scripts to {args.sweep_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", type=str, default="./deep-interact")
    parser.add_argument("--sweep_dir", type=str)
    parser.add_argument("--job_name", type=str)

    args = parser.parse_args()
    main(args)
