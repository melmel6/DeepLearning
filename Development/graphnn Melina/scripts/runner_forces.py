import argparse
import itertools
import json
import logging
import math
import os
import sys

import numpy as np
import torch

from context import graphnn
from graphnn import data
from graphnn import model_forces, model_painn


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Train graph convolution network", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Load model parameters from previous run",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="Atomic interaction cutoff distance [Å]",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Train/test/validation split file json",
    )
    parser.add_argument(
        "--num_interactions",
        type=int,
        default=3,
        help="Number of interaction layers used",
    )
    parser.add_argument(
        "--use_painn_model",
        action="store_true",
        help="Use PaiNN model rather than Schnet (w edge)",
    )
    parser.add_argument(
        "--node_size", type=int, default=64, help="Size of hidden node states"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/model_output",
        help="Path to output directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/qm9.db",
        help="Path to ASE database",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=int(1e6),
        help="Maximum number of optimisation steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Set which device to use for training e.g. 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--update_edges",
        action="store_true",
        help="Enable edge updates in model",
    )
    parser.add_argument(
        "--atomwise_normalization",
        action="store_true",
        help="Enable normalization on atom-level rather than on global graph output",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of molecules per minibatch",
    )

    parser.add_argument(
        "--initial_lr",
        type=float,
        default=0.0001,
        help="Initial learning rate",
    )

    parser.add_argument(
        "--forces_property",
        type=str,
        default="forces",
        help="Name of forces property in ASE database",
    )
    parser.add_argument(
        "--energy_property",
        type=str,
        default="energy",
        help="Name of energy property in ASE database",
    )

    parser.add_argument(
        "--forces_weight",
        type=float,
        default=0.5,
        help="Tradeoff between training on forces (weight=1) and energy (weight=0)",
    )

    parser.add_argument(
        "--direct_force_output",
        action="store_true",
        help="Calculate forces directly with neural network instead of taking the gradient",
    )

    return parser.parse_args(arg_list)


def split_data(dataset, args):
    # Load or generate splits
    if args.split_file:
        with open(args.split_file, "r") as fp:
            splits = json.load(fp)
    else:
        datalen = len(dataset)
        num_validation = int(math.ceil(datalen * 0.10))
        indices = np.random.permutation(len(dataset))
        splits = {
            "train": indices[num_validation:].tolist(),
            "validation": indices[:num_validation].tolist(),
        }

    # Save split file
    with open(os.path.join(args.output_dir, "datasplits.json"), "w") as f:
        json.dump(splits, f)

    # Split the dataset
    datasplits = {}
    for key, indices in splits.items():
        datasplits[key] = torch.utils.data.Subset(dataset, indices)
    return datasplits


def eval_model(model, dataloader, device, forces_weight):
    energy_running_ae = 0
    energy_running_se = 0
    energy_running_count = 0

    forces_running_l2_ae = 0
    forces_running_l2_se = 0
    forces_running_c_ae = 0
    forces_running_c_se = 0
    forces_running_count = 0
    forces_running_loss = 0

    running_loss = 0
    running_loss_count = 0
    criterion = torch.nn.MSELoss()

    for batch in dataloader:
        device_batch = {
            k: v.to(device=device, non_blocking=True) for k, v in batch.items()
        }
        out = model(device_batch)

        forces_loss = forces_criterion(
            out["forces"], device_batch["forces"], device_batch["num_nodes"]
        ).item()
        energy_loss = criterion(out["energy"], device_batch["energy"])
        total_loss = forces_weight * forces_loss + (1 - forces_weight) * energy_loss
        running_loss += total_loss.item() * batch["energy"].shape[0]
        running_loss_count += batch["energy"].shape[0]

        outputs = {key: val.detach().cpu().numpy() for key, val in out.items()}

        energy_targets = batch["energy"].detach().cpu().numpy()
        energy_running_ae += np.sum(np.abs(energy_targets - outputs["energy"]), axis=0)
        energy_running_se += np.sum(
            np.square(energy_targets - outputs["energy"]), axis=0
        )
        energy_running_count += energy_targets.shape[0]

        forces_targets = batch["forces"].detach().cpu().numpy()
        forces_diff = forces_targets - outputs["forces"]
        forces_l2_norm = np.sqrt(np.sum(np.square(forces_diff), axis=2))

        forces_running_c_ae += np.sum(np.abs(forces_diff))
        forces_running_c_se += np.sum(np.square(forces_diff))

        forces_running_l2_ae += np.sum(np.abs(forces_l2_norm))
        forces_running_l2_se += np.sum(np.square(forces_l2_norm))
        forces_running_count += np.sum(batch["num_nodes"].detach().cpu().numpy())

    energy_mae = energy_running_ae / energy_running_count
    energy_rmse = np.sqrt(energy_running_se / energy_running_count)

    forces_l2_mae = forces_running_l2_ae / forces_running_count
    forces_l2_rmse = np.sqrt(forces_running_l2_se / forces_running_count)

    forces_c_mae = forces_running_c_ae / (forces_running_count * 3)
    forces_c_rmse = forces_running_c_se / (forces_running_count * 3)

    total_loss = running_loss / running_loss_count

    evaluation = {
        "energy_mae": energy_mae,
        "energy_rmse": energy_rmse,
        "forces_l2_mae": forces_l2_mae,
        "forces_l2_rmse": forces_l2_rmse,
        "forces_mae": forces_c_mae,
        "forces_rmse": forces_c_rmse,
        "sqrt(total_loss)": np.sqrt(total_loss),
    }

    return evaluation


def get_normalization(dataset, per_atom=True):
    try:
        num_targets = len(dataset.transformer.targets)
    except AttributeError:
        num_targets = 1
    # Use double precision to avoid overflows
    x_sum = torch.zeros(num_targets, dtype=torch.double)
    x_2 = torch.zeros(num_targets, dtype=torch.double)
    num_objects = 0
    for i, sample in enumerate(dataset):
        if i == 0:
            # Estimate "bias" from 1 sample
            # to avoid overflows for large valued datasets
            if per_atom:
                bias = sample["energy"] / sample["num_nodes"]
            else:
                bias = sample["energy"]
        x = sample["energy"]
        if per_atom:
            x = x / sample["num_nodes"]
        x -= bias
        x_sum += x
        x_2 += x ** 2.0
        num_objects += 1
    # Var(X) = E[X^2] - E[X]^2
    x_mean = x_sum / num_objects
    x_var = x_2 / num_objects - x_mean ** 2.0
    x_mean = x_mean + bias

    default_type = torch.get_default_dtype()

    return x_mean.type(default_type), torch.sqrt(x_var).type(default_type)


def get_model(args, **kwargs):
    if args.use_painn_model:
        if args.update_edges:
            raise Warning("update_edges argument ignored for PaiNN model")
        net = model_painn.PainnModel(
            args.num_interactions,
            args.node_size,
            args.cutoff,
            normalize_atomwise=args.atomwise_normalization,
            direct_force_output=args.direct_force_output,
            **kwargs
        )
    else:
        net = model_forces.SchnetModelForces(
            args.num_interactions,
            args.node_size,
            args.cutoff,
            update_edges=args.update_edges,
            normalize_atomwise=args.atomwise_normalization,
            **kwargs
        )
    return net


def forces_criterion(predicted, target, node_count, reduction="mean"):
    # predicted, target are (bs, max_nodes, 3) tensors
    # node_count is (bs) tensor
    diff = predicted - target
    total_squared_norm = torch.sum(torch.sum(torch.square(diff), axis=2), axis=1)  # bs
    assert len(node_count.shape) == 1
    avg_squared_norm = total_squared_norm / node_count
    if reduction == "mean":
        scalar = torch.mean(avg_squared_norm)
    elif reduction == "sum":
        scalar = torch.sum(avg_squared_norm)
    else:
        raise ValueError("Reduction must be 'mean' or 'sum'")
    return scalar


def main():
    args = get_arguments()

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "printlog.txt"), mode="w"
            ),
            logging.StreamHandler(),
        ],
    )

    # Save command line args
    with open(os.path.join(args.output_dir, "commandline_args.txt"), "w") as f:
        f.write("\n".join(sys.argv[1:]))
    # Save parsed command line arguments
    with open(os.path.join(args.output_dir, "arguments.json"), "w") as f:
        json.dump(vars(args), f)

    # Create device
    device = torch.device(args.device)
    # Put a tensor on the device before loading data
    # This way the GPU appears to be in use when other users run gpustat
    torch.tensor([0], device=device)

    # Setup dataset and loader
    logging.info("loading data %s", args.dataset)
    dataset = data.AseDbData(
        args.dataset,
        data.TransformRowToGraphXyz(
            cutoff=args.cutoff,
            energy_property=args.energy_property,
            forces_property=args.forces_property,
        ),
    )
    dataset = data.BufferData(dataset)  # Load data into host memory

    # Split data into train and validation sets
    datasplits = split_data(dataset, args)
    logging.info("Computing mean and variance")
    target_mean, target_stddev = get_normalization(
        datasplits["train"], per_atom=args.atomwise_normalization
    )
    logging.debug("target_mean=%f, target_stddev=%f" % (target_mean, target_stddev))

    # Setup loaders
    train_loader = torch.utils.data.DataLoader(
        datasplits["train"],
        args.batch_size,
        sampler=torch.utils.data.RandomSampler(datasplits["train"]),
        collate_fn=data.CollateAtomsdata(pin_memory=device.type == "cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        datasplits["validation"],
        args.batch_size,
        collate_fn=data.CollateAtomsdata(pin_memory=device.type == "cuda"),
    )

    # Initialise model
    net = get_model(args, target_mean=target_mean, target_stddev=target_stddev)
    net = net.to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.initial_lr)
    criterion = torch.nn.MSELoss()
    scheduler_fn = lambda step: 0.96 ** (step / 100000)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_fn)

    log_interval = 5000
    running_loss = 0
    running_loss_count = 0
    best_val_loss = np.inf
    step = 0
    # Restore checkpoint
    if args.load_model:
        state_dict = torch.load(args.load_model)
        net.load_state_dict(state_dict["model"])
        step = state_dict["step"]
        best_val_loss = state_dict["best_val_loss"]
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])

    logging.info("start training")
    for epoch in itertools.count():
        for batch_host in train_loader:
            # Transfer to 'device'
            batch = {
                k: v.to(device=device, non_blocking=True)
                for (k, v) in batch_host.items()
            }

            # Reset gradient
            optimizer.zero_grad()

            # Forward, backward and optimize
            outputs = net(
                batch, compute_stress=False, compute_forces=bool(args.forces_weight)
            )
            energy_loss = criterion(outputs["energy"], batch["energy"])
            if args.forces_weight:
                forces_loss = forces_criterion(
                    outputs["forces"], batch["forces"], batch["num_nodes"]
                )
            else:
                forces_loss = 0.0
            total_loss = (
                args.forces_weight * forces_loss
                + (1 - args.forces_weight) * energy_loss
            )
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * batch["energy"].shape[0]
            running_loss_count += batch["energy"].shape[0]

            # print(step, loss_value)
            # Validate and save model
            if (step % log_interval == 0) or ((step + 1) == args.max_steps):
                train_loss = running_loss / running_loss_count
                running_loss = running_loss_count = 0

                eval_dict = eval_model(net, val_loader, device, args.forces_weight)
                eval_formatted = ", ".join(
                    ["%s=%g" % (k, v) for (k, v) in eval_dict.items()]
                )

                logging.info(
                    "step=%d, %s, sqrt(train_loss)=%g",
                    step,
                    eval_formatted,
                    math.sqrt(train_loss),
                )

                # Save checkpoint
                if eval_dict["sqrt(total_loss)"] < best_val_loss:
                    best_val_loss = eval_dict["sqrt(total_loss)"]
                    torch.save(
                        {
                            "model": net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "step": step,
                            "best_val_loss": best_val_loss,
                        },
                        os.path.join(args.output_dir, "best_model.pth"),
                    )
            step += 1

            scheduler.step()

            if step >= args.max_steps:
                logging.info("Max steps reached, exiting")
                torch.save(
                    {
                        "model": net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step,
                        "best_val_loss": best_val_loss,
                    },
                    os.path.join(args.output_dir, "exit_model.pth"),
                )
                sys.exit(0)


if __name__ == "__main__":
    main()
