import pdb
from Geom3D.datasets import MoleculeDatasetQM9, Molecule3DDataset

data_root = "../data/molecule_datasets/{}".format("QM9")
dataset_tuning = MoleculeDatasetQM9(
    data_root,
    dataset="QM9",
    task="all",
    rotation_transform=None,
)

dat_root = "/home/patrick/data/QM9"
dataset_pretraining = Molecule3DDataset(
    data_root,
    "QM9",
    mask_ratio=0,
    remove_center=True,
    use_extend_graph=True,
    transform=None,
)

pdb.set_trace()
