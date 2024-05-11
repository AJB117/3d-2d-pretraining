import sys
import os
import yaml
import argparse
from email.policy import default

parser = argparse.ArgumentParser()

# about seed and basic info
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=int, default=0)

parser.add_argument(
    "--model_3d",
    type=str,
    default="SchNet",
    choices=["SchNet", "PaiNN", "EGNN"],
)
parser.add_argument(
    "--model_2d",
    type=str,
    default="GIN",
    choices=["GIN", "GAT", "GCN", "GraphSAGE", "Transformer", "GT", "GPS"],
)

# about dataset and dataloader
parser.add_argument("--dataset", type=str, default="QM9")
parser.add_argument("--task", type=str)
parser.add_argument("--num_workers", type=int, default=0)

# for MD17
# The default hyper from here: https://github.com/divelab/DIG_storage/tree/main/3dgraph/MD17
parser.add_argument("--MD17_energy_coeff", type=float, default=0.05)
parser.add_argument("--MD17_force_coeff", type=float, default=0.95)
parser.add_argument(
    "--energy_force_with_normalization",
    dest="energy_force_with_normalization",
    action="store_true",
)
parser.add_argument(
    "--energy_force_no_normalization",
    dest="energy_force_with_normalization",
    action="store_false",
)
parser.set_defaults(energy_force_with_normalization=False)

# about training strategies
parser.add_argument(
    "--split",
    type=str,
    default="customized_01",
    choices=["customized_01", "customized_02", "random", "50k_split"],
)
parser.add_argument("--MD17_train_batch_size", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--lr_scale", type=float, default=1)
parser.add_argument("--decay", type=float, default=0)
parser.add_argument("--print_every_epoch", type=int, default=1)
parser.add_argument("--loss", type=str, default="mae", choices=["mse", "mae"])
parser.add_argument("--lr_scheduler", type=str, default="CosineAnnealingLR")
parser.add_argument("--lr_decay_factor", type=float, default=0.5)
parser.add_argument("--lr_decay_step_size", type=int, default=100)
parser.add_argument("--lr_decay_patience", type=int, default=50)
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--StepLRCustomized_scheduler", type=int, nargs="+", default=[150])
parser.add_argument("--verbose", dest="verbose", action="store_true")
parser.add_argument("--no_verbose", dest="verbose", action="store_false")
parser.set_defaults(verbose=False)
parser.add_argument(
    "--use_rotation_transform", dest="use_rotation_transform", action="store_true"
)
parser.add_argument(
    "--no_rotation_transform", dest="use_rotation_transform", action="store_false"
)
parser.set_defaults(use_rotation_transform=False)

# for SchNet
parser.add_argument("--SchNet_num_filters", type=int, default=128)
parser.add_argument("--SchNet_num_interactions", type=int, default=6)
parser.add_argument("--SchNet_num_gaussians", type=int, default=51)
parser.add_argument("--SchNet_cutoff", type=float, default=10)
parser.add_argument(
    "--SchNet_readout", type=str, default="mean", choices=["mean", "add"]
)
parser.add_argument("--SchNet_gamma", type=float, default=None)

######################### for GraphMVP SSL #########################
### for 2D GNN
parser.add_argument("--gnn_type", type=str, default="GIN")
parser.add_argument("--num_layer", type=int, default=5)
parser.add_argument("--emb_dim", type=int, default=300)
parser.add_argument("--dropout_ratio", type=float, default=0.0)
parser.add_argument("--graph_pooling", type=str, default="mean")
parser.add_argument("--JK", type=str, default="last")
parser.add_argument("--gnn_2d_lr_scale", type=float, default=1)

parser.add_argument(
    "--alpha_1", type=float, default=1, help="balancing hyperparam for GraphMVP"
)
parser.add_argument(
    "--alpha_2", type=float, default=0.1, help="balancing hyperparam for GraphMVP"
)
parser.add_argument("--beta", type=float, default=1, help="VAE hyperparam for GraphMVP")
parser.add_argument("--AE_loss", type=str, default="l2", help="loss for AE in GraphMVP")
parser.add_argument("--AE_model", type=str, default="VAE", help="AE model for GraphMVP")
parser.add_argument("--detach_target", dest="detach_target", action="store_true")
parser.add_argument("--gmvp_gnn_lr_scale", type=float, default=1)
parser.add_argument("--gmvp_schnet_lr_scale", type=float, default=0.1)


### for 3D GNN
parser.add_argument("--gnn_3d_lr_scale", type=float, default=1)

### for masking
# parser.add_argument("--SSL_masking_ratio", type=float, default=0.15) # for prev methods
parser.add_argument("--SSL_masking_ratio", type=float, default=0)

parser.add_argument("--T", type=float, default=0.1)
parser.add_argument("--normalize", dest="normalize", action="store_true")
parser.add_argument("--no_normalize", dest="normalize", action="store_false")


# This is only for 3D to 2D
parser.add_argument("--use_extend_graph", dest="use_extend_graph", action="store_true")
parser.add_argument("--no_extend_graph", dest="use_extend_graph", action="store_false")
parser.set_defaults(use_extend_graph=True)
# This is only for 2D to 3D
parser.add_argument("--noise_on_one_hot", dest="noise_on_one_hot", action="store_true")
parser.add_argument(
    "--no_noise_on_one_hot", dest="noise_on_one_hot", action="store_false"
)
parser.set_defaults(noise_on_one_hot=True)
parser.add_argument("--SDE_anneal_power", type=float, default=0)
# This is only for 2D to 3D to MoleculeNet property
parser.add_argument("--molecule_property_SDE_2D", type=float, default=1)

##### about if we would print out eval metric for training data
parser.add_argument("--eval_train", dest="eval_train", action="store_true")
parser.add_argument("--no_eval_train", dest="eval_train", action="store_false")
parser.set_defaults(eval_train=False)

parser.add_argument("--eval_test", dest="eval_test", action="store_true")
parser.add_argument("--no_eval_test", dest="eval_test", action="store_false")
parser.set_defaults(eval_test=True)

parser.add_argument("--input_data_dir", type=str, default="")

# about loading and saving
parser.add_argument("--input_model_file", type=str, default="")
parser.add_argument("--output_model_dir", type=str, default="")

parser.add_argument("--threshold", type=float, default=0)

parser.add_argument(
    "--pretrain_checkpoint_path",
    type=str,
    help="path of pretraining checkpoint to load",
)
parser.add_argument("--mode", type=str, default="method", help="baseline to use")
parser.add_argument(
    "--use_generated_dataset", dest="use_generated_dataset", action="store_true"
)

### OUR METHOD, NOT MOLECULESDE/GRAPHMVP
parser.add_argument("--input_model_file_3d", type=str, default="")

# GNN_pos
parser.add_argument("--num_layer_pos", type=int, default=3)
parser.add_argument("--emb_dim_pos", type=int, default=64)
parser.add_argument("--dropout_ratio_pos", type=float, default=0.2)
parser.add_argument("--gnn_type_pos", type=str, default="GIN")
parser.add_argument("--JK_pos", type=str, default="last")
parser.add_argument("--gnn_2d_pos_lr_scale", type=float, default=1)

# EGNN
parser.add_argument("--emb_dim_egnn", type=int, default=300)
parser.add_argument("--n_layers_egnn", type=int, default=5)
parser.add_argument("--positions_weight_egnn", type=float, default=1.0)
parser.add_argument("--attention_egnn", action="store_true")
parser.add_argument(
    "--require_3d",
    action="store_true",
    help="Require 3D data; PCQM4Mv2 only has 3D information for a subset of the dataset",
)
parser.add_argument("--output_model_name", type=str, default="")
parser.add_argument("--process_num", type=int, default=1)

# DEEP INTERACTION HYPERPARAMS
parser.add_argument(
    "--interaction_rep_2d",
    type=str,
    default="vnode",
    choices=["sum", "mean", "vnode"],
    help="how to represent the interaction for 2D GNN",
)
parser.add_argument(
    "--interaction_rep_3d",
    type=str,
    default="com",
    choices=["sum", "mean", "com", "const_radius", ""],
    help="how to represent the interaction for 3D GNN, com: center of mass, const_radius: hardcoded constant radius. Use '' for baseline 3D models to avoid generating extra 3d coords",
)
parser.add_argument(
    "--interaction_agg",
    type=str,
    choices=["cat", "sum", "mean"],
    default="cat",
    help="how to aggregate the interactions",
)
parser.add_argument(
    "--num_interaction_blocks", type=int, default=6, help="number of interaction blocks"
)
parser.add_argument(
    "--final_pool",
    choices=["cat", "mean", "attention"],
    default="cat",
    help="how to pool the final embeddings + interaction emeddings",
)
parser.add_argument("--residual", action="store_true", help="add residual connections")
parser.add_argument(
    "--layer_norm", action="store_true", help="use layer norm over batch norm"
)
parser.add_argument(
    "--batch_norm", action="store_true", help="use batch norm over layer norm"
)
parser.add_argument("--gat_heads", type=int, default=4, help="number of GAT heads")
parser.add_argument(
    "--initialization",
    type=str,
    default="glorot_uniform",
    choices=["glorot_uniform", "he_uniform", "glorot_normal", "he_normal"],
    help="initialization",
)
parser.add_argument(
    "--pretrain_2d_tasks",
    type=str,
    nargs="+",
    default=["interatomic_dist", "bond_angle"],
    help="tasks to pretrain the 2D blocks on, ordered by block",
)
parser.add_argument(
    "--pretrain_3d_tasks",
    type=str,
    nargs="+",
    default=["edge_existence", "edge_classification"],
    help="tasks to pretrain the 3D blocks on, ordered by block",
)
parser.add_argument(
    "--pretrain_2d_task_indices",
    type=int,
    nargs="+",
    default=[0, 1],
    help="indices of blocks for 2D pretraining tasks. [i, j] means start first task at block i and second task at block j.",
)
parser.add_argument(
    "--pretrain_3d_task_indices",
    type=int,
    nargs="+",
    default=[0, 1],
    help="indices of blocks for 3D pretraining tasks. [i, j] means start first task at block i and second task at block j.",
)
parser.add_argument(
    "--pretrain_2d_balances",
    type=float,
    nargs="+",
    default=[1.0, 1.0],
    # default=[1e-3, 1.0],
    help="balancing parameters for 2D blocks' tasks so that all losses are on the same scale",
)
parser.add_argument(
    "--pretrain_3d_balances",
    type=float,
    nargs="+",
    default=[1.0, 1.0],
    # default=[1e-1, 10.0],
    help="balancing parameters for 3D blocks' tasks so that all losses are on the same scale",
)
parser.add_argument(
    "--pretrain_strategy",
    choices=["geometric", "masking", "both"],
    default="geometric",
    help="pretraining strategy",
)
parser.add_argument(
    "--transformer_heads",
    default=4,
    type=int,
    help="for transformer conv for the 2d model",
)
parser.add_argument("--wandb", action="store_true", help="use wandb")
parser.add_argument(
    "--use_3d_only",
    action="store_true",
    help="use 3d layers only",
)
parser.add_argument(
    "--use_2d_only",
    action="store_true",
    help="use 2d layers only",
)
parser.add_argument(
    "--pretrain_interatomic_samples",
    type=int,
    help="number of samples for pretraining for interatomic distances and edge existence",
    default=-1,
)
parser.add_argument(
    "--pretrain_neg_link_samples",
    type=int,
    help="number of samples for pretraining for interatomic distances and edge existence",
    default=50,
)
parser.add_argument(
    "--config_dir",
    type=str,
    default="./deep-interact/configs",
    help="dir to save config file",
)
parser.add_argument(
    "--config_name",
    type=str,
    default="",
    help="name of config file, leave blank if you want to use manually set args",
)
parser.add_argument("--save_config", action="store_true", help="save config file")
parser.add_argument(
    "--diff_interactor_per_block",
    action="store_true",
    help="use a different interactor MLP per block",
)
parser.add_argument(
    "--interactor_residual",
    action="store_true",
    help="use residual for interaction blocks",
)
parser.add_argument("--interactor_activation", default="Swish")
parser.add_argument(
    "--mixup_ratio", default=0.25, type=float, help="ratio for mixup interaction"
)
parser.add_argument(
    "--transfer",
    action="store_true",
    help="only for 2d-only usage. adds 3d atom embeddings to the 2d atom embeddings",
)
parser.add_argument(
    "--all_losses_at_end",
    action="store_true",
    help="ablate with all loss functions at the end of the blocks",
)
parser.add_argument(
    "--weight_sharing",
    action="store_true",
    help="weight sharing of message-passing layers prior to interactions",
)
parser.add_argument("--interact_every_block", action="store_true")

parser.add_argument("--rep_type", choices=["atom", "bond"], default="atom")
parser.add_argument(
    "--mix_embs_pretrain",
    action="store_true",
    help="mix embeddings when predicting pretraining features",
)
parser.add_argument(
    "--use_tanh_dihedral",
    action="store_true",
    help="use tanh activaiton for dihedral angle prediction",
)
parser.add_argument(
    "--use_shallow_predictors",
    action="store_true",
    help="use shallow linear predictors for pretraining, ow use 2-layer MLPs",
)
parser.add_argument("--pretrain_decoding", action="store_true")
parser.add_argument(
    "--symm_dihedrals",
    action="store_true",
    help="symmetrize dihedral angles for pretraining",
)
parser.add_argument(
    "--classify_dihedrals",
    action="store_true",
    help="treat dihedral prediction as classification task; 20 bins without symm, 10 bins with symm",
)

args = parser.parse_args()

if args.config_dir and args.config_name:
    with open(os.path.join(args.config_dir, args.config_name), "r") as f:
        argdict = yaml.load(f, yaml.FullLoader)
        for key in argdict:
            setattr(args, key, argdict[key])

elif args.save_config:
    if not os.path.exists(args.config_dir):
        os.makedirs(args.config_dir)

    args.config_name = args.output_model_name

    if not args.config_name.endswith(".yml"):
        args.config_name = args.config_name + ".yml"

        flags = ""
        for k, v in vars(args).items():
            if k in ["config_dir", "config_name", "save_config"]:
                continue
            if isinstance(v, bool):
                if not v:
                    continue
                flags += f"--{k} "

            elif isinstance(v, list):
                flags += f"--{k} {' '.join(map(str, v))} "
            else:
                flags += f"--{k} {v} "

        args.flags = flags

    config_path = os.path.join(args.config_dir, args.dataset + "_" + args.config_name)

    with open(config_path, "w") as f:
        args_dict = vars(args)
        yaml.dump(args_dict, f)

    print(config_path)

if not args.save_config:
    print("arguments\t", args)
