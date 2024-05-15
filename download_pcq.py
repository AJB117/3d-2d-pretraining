import os
import boto3

s3 = boto3.client("s3")
BUCKET = "3d-2d-pretraining"

prefix = "PCQM4Mv2"

path = f"./data/{prefix}/processed"
os.makedirs(path, exist_ok=True)
path = f"./data/{prefix}/raw"
os.makedirs(path, exist_ok=True)
path = f"./data/{prefix}"

for file in [
    "geometric_data_processed.pt",
    "data.csv.gz",
    "pre_filter.pt",
    "pre_transform.pt",
    "smiles.csv",
]:
    print(f"downloading {path}/{file}")
    s3.download_file(BUCKET, file, f"{path}/{file}")

    # s3.download_file(BUCKET, file, f"./data/{file}")
