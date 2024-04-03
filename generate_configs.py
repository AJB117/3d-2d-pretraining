import os
import argparse
import yaml


def main(args):
    experiment_dir = os.path.join(args.base_dir, args.experiment_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    config_file = args.config_file_base

    hparam_values = args.hparam_choices
    for hparam_value in hparam_values:
        config = yaml.load(open(config_file), Loader=yaml.FullLoader)
        if args.hparam not in config:
            raise ValueError(f"{args.hparam} not in config file")

        config[args.hparam] = hparam_value
        new_config_name = (
            f"{config['config_name'].split('.yml')[0]}_{args.hparam}_{hparam_value}.yml"
        )
        config["config_name"] = new_config_name
        with open(os.path.join(experiment_dir, new_config_name), "w") as f:
            yaml.dump(config, f)

    print(experiment_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./deep-interact")
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--config_file_base", type=str)
    parser.add_argument("--hparam", type=str, default="task")
    parser.add_argument(
        "--hparam_choices",
        type=str,
        nargs="+",
        default=[
            "mu",
            "alpha",
            "gap",
            "homo",
            "lumo",
            "r2",
            "zpve",
            "u0",
            "u298",
            "h298",
            "g298",
            "cv",
        ],
    )

    args = parser.parse_args()
    main(args)
