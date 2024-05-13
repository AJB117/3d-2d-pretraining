import pdb
from Geom3D.datasets import MoleculeDatasetQM8

data_root = "data/molecule_datasets/QM8"

dataset = MoleculeDatasetQM8(
    data_root,
    dataset="QM8",
    task="E1-CC2",
    rotation_transform=None,
    transform=None,
    use_pure_atomic_num=False,
)

pdb.set_trace()