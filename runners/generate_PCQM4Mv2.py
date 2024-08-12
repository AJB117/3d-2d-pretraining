from Geom3D.datasets import PCQM4Mv2


if __name__ == "__main__":
    # root_dir = "/home/patrick/data/PCQM4Mv2-pretraining"
    # test_dataset = PCQM4Mv2(root=root_dir, transform=None, pretraining=True)
    root_dir = "/home/zqe3cg/3d-2d-pretraining/data/PCQM4Mv2"
    dataset = PCQM4Mv2(root=root_dir, transform=None, pretraining=False)
