import pandas as pd
import torch
import os
import argparse
from collections import defaultdict


def main(args):
    data = defaultdict(list)

    path = os.path.join(args.root_dir, args.dataset)
    for file in os.listdir(path):
        results = torch.load(os.path.join(path, file))
        task = results["task"]

        result_metrics = [
            results[f"train_mae_{task}"],
            results[f"val_mae_{task}"],
            results[f"test_mae_{task}"],
        ]

        data["task"].append(task)
        data["train_mae"].append(result_metrics[0])
        data["val_mae"].append(result_metrics[1])
        data["test_mae"].append(result_metrics[2])

    pd.DataFrame(data).to_csv(
        os.path.join(path, "results.csv"), header=data.keys(), index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--root_dir")
    main(parser.parse_args())
