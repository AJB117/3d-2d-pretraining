import os
import boto3

s3 = boto3.client("s3")
BUCKET = "3d-2d-pretraining"

prefix = "QM9"

path = f"./data/molecule_datasets/{prefix}/processed"
os.makedirs(path, exist_ok=True)
path = f"./data/molecule_datasets/{prefix}/raw"
os.makedirs(path, exist_ok=True)

path = "./data/molecule_datasets"

sub_prefixes = ["processed", "raw"]
for sub_prefix in sub_prefixes:
    files = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{prefix}/{sub_prefix}").get(
        "Contents"
    )
    for file in files:
        file = file.get("Key")
        print(f"downloading {path}/{file}")
        s3.download_file(BUCKET, file, f"{path}/{file}")

    # s3.download_file(BUCKET, file, f"./data/{file}")
