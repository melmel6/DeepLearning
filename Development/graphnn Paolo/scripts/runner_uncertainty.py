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
from graphnn import model_uncertainty


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
        "--node_size", type=int, default=64, help="Size of hidden node states"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/model_uncertainty_output",
        help="Path to output directory",
    )
    parser.add_argument(
        "--dataset", type=str, default="data/qm9.db", help="Path to ASE database",
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
        "--update_edges", action="store_true", help="Enable edge updates in model",
    )
    parser.add_argument(
        "--atomwise_normalization",
        action="store_true",
        help="Enable normalization on atom-level rather than on global graph output",
    )
    parser.add_argument(
        "--target", type=str, default="U0", help="Name of target property",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay",
    )
    parser.add_argument(
        "--scale_transform",
        type=str,
        default="softplus",
        help="Set function used to transform the scale output: 'softplus' or 'exp'",
    )
    parser.add_argument(
        "--loglik",
        type=str,
        default="normal",
        help="Set loglik function used to compute the loss: 'normal' or 'cauchy'",
    )
    parser.add_argument(
        "--interp_steps",
        type=int,
        default=1,
        help="Loss MSE to NLL interpolation steps.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Loss MSE warmup steps.",
    )
    parser.add_argument(
        "--bootstrap", action="store_true", help="Bootstrap sample training data.",
    )
    parser.add_argument(
        "--predict", action="store_true",
        help="Predict on all data splits with best model after training.",
    )
    parser.add_argument(
        "--interp_scale", type=float, default=0.1,
        help="Scale interpolation starting value.",
    )

    return parser.parse_args(arg_list)


def load_dataset(args):
    """Load dataset."""
    logging.info("Loading data: %s", args.dataset)
    dataset = data.AseDbData(
        args.dataset, data.TransformRowToGraph(cutoff=args.cutoff, targets=args.target)
    )
    dataset = data.BufferData(dataset)  # Load data into host memory
    return dataset


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


def bootstrap_data(dataset, args):
    # Generate bootsrap sample indices
    n = len(dataset)
    indices = torch.randint(0, n, [n])
    # Save bootstrap indices file
    with open(os.path.join(args.output_dir, "bootstrap.json"), "w") as f:
        json.dump({"bootstrap": indices.tolist()}, f)
    # Sample dataset
    dataset = torch.utils.data.Subset(dataset, indices)
    return dataset


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
                bias = sample["targets"] / sample["num_nodes"]
            else:
                bias = sample["targets"]
        x = sample["targets"]
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
    net = model_uncertainty.SchnetModel(
        num_interactions=args.num_interactions,
        hidden_state_size=args.node_size,
        cutoff=args.cutoff,
        update_edges=args.update_edges,
        normalize_atomwise=args.atomwise_normalization,
        scale_transform=args.scale_transform,
        **kwargs
    )
    return net


@torch.no_grad()
def eval_model(model, dataloader, device, nll_function):
    running_nll = 0
    running_ae = 0
    running_se = 0
    running_scale = 0
    running_count = 0
    for batch in dataloader:
        batch = {
            k: v.to(device=device, non_blocking=True) for k, v in batch.items()
        }
        loc, scale = model(batch)
        targets = batch["targets"]
        running_nll += nll_function(loc, scale, targets, reduction="sum")
        running_ae += torch.sum(torch.abs(targets - loc))
        running_se += torch.sum(torch.square(targets - loc))
        running_scale += torch.sum(scale)
        running_count += targets.shape[0]
    nll = running_nll / running_count
    mae = running_ae / running_count
    rmse = torch.sqrt(running_se / running_count)
    mscale = running_scale / running_count
    return nll.item(), mae.item(), rmse.item(), mscale.item()


# Constants for computing log likelihood
LOGPI = np.log(np.pi)
LOG2PI = np.log(2 * np.pi)


def normal_nll(loc, scale, targets, reduction="mean"):
    """Normal negative log likelihood function.

    The location (loc) specifies the mean. The scale specifies the variance.
    """
    loglik = -0.5 * (torch.log(scale) + LOG2PI + (targets - loc)**2 / scale)
    return _apply_reduction(loglik, reduction)


def cauchy_nll(loc, scale, targets, reduction="mean"):
    """Cauchy negative log likelihood function."""
    loglik = torch.log(scale) - LOGPI - torch.log((targets - loc)**2 + scale**2)
    return _apply_reduction(loglik, reduction)


def _apply_reduction(loglik, reduction):
    if reduction == "mean":
        return -torch.mean(loglik)
    elif reduction == "sum":
        return -torch.sum(loglik)
    elif reduction in ["none", None]:
        return -loglik
    assert reduction in ["mean", "sum", "none", None]


class Interpolator:

    def __init__(self, interp_steps, warmup_steps=0):
        self.interp_steps = interp_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        lam = (step - self.warmup_steps) / self.interp_steps
        return 1 - min(max(lam, 0.0), 1.0)


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

    # Load dataset
    dataset = load_dataset(args)

    # Split data into train and validation sets
    datasplits = split_data(dataset, args)

    # Train model
    train(args, device, datasplits)

    # Predict with model
    if args.predict:
        predict(args, device, datasplits)

    logging.info("Done.")


def train(args, device, datasplits):
    """Train network."""

    # Bootstrap sample the training data
    if args.bootstrap:
        datasplits["train"] = bootstrap_data(datasplits["train"], args)

    logging.info("Computing mean and variance.")
    target_mean, target_stddev = get_normalization(
        datasplits["train"], per_atom=args.atomwise_normalization
    )
    logging.debug("target_mean=%f, target_stddev=%f" % (target_mean, target_stddev))

    # Setup loaders
    train_loader = torch.utils.data.DataLoader(
        datasplits["train"],
        100,
        sampler=torch.utils.data.RandomSampler(datasplits["train"]),
        collate_fn=data.CollateAtomsdata(pin_memory=device.type == "cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        datasplits["validation"],
        32,
        collate_fn=data.CollateAtomsdata(pin_memory=device.type == "cuda"),
    )

    # Initialise model
    net = get_model(args, target_mean=target_mean, target_stddev=target_stddev)
    net = net.to(device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.96 ** (step / 100000)
    )

    # Setup loss functions
    lossf_mse = torch.nn.MSELoss()
    assert args.loglik in ["normal", "cauchy"]
    if args.loglik == "normal":
        lossf_nll = normal_nll
    elif args.loglik == "cauchy":
        lossf_nll = cauchy_nll

    # Setup loss function interpolator
    interp = Interpolator(
        interp_steps=args.interp_steps,
        warmup_steps=args.warmup_steps
    )

    log_interval = 10000
    running_loss = 0
    running_nll = 0
    running_count = 0
    best_val_nll = np.inf
    step = 0
    # Restore checkpoint
    if args.load_model:
        state_dict = torch.load(args.load_model)
        net.load_state_dict(state_dict["model"])
        step = state_dict["step"]
        best_val_nll = state_dict["best_val_nll"]
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])

    logging.info("Start training...")
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
            loc, scale = net(batch)
            lam1 = interp(step)  # From 1 to 0
            lam2 = args.interp_scale + \
                (1 - args.interp_scale) * (1 - lam1)  # From interp_scale to 1
            loss_mse = lossf_mse(loc, batch["targets"])
            loss_nll = lossf_nll(loc, lam2 * scale, batch["targets"])
            loss = lam1 * loss_mse + (1 - lam1) * loss_nll
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch["targets"].shape[0]
            running_nll += loss_nll.item() * batch["targets"].shape[0]
            running_count += batch["targets"].shape[0]

            # Validate and save model
            if (step % log_interval == 0) or ((step + 1) == args.max_steps):
                train_loss = running_loss / running_count
                train_nll = running_nll / running_count
                running_loss = running_nll = running_count = 0

                val_nll, val_mae, val_rmse, val_mscale = eval_model(
                    net, val_loader, device, lossf_nll
                )

                logging.info(
                    "step=%d, train_loss=%g, train_nll=%g, val_nll=%g, val_mae=%g, val_rmse=%g, val_mscale=%g",
                    step,
                    train_loss,
                    train_nll,
                    val_nll,
                    val_mae,
                    val_rmse,
                    val_mscale,
                )

                # Save checkpoint
                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    torch.save(
                        {
                            "model": net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "step": step,
                            "best_val_nll": best_val_nll,
                        },
                        os.path.join(args.output_dir, "best_model.pth"),
                    )

            step += 1
            scheduler.step()

            if step >= args.max_steps:
                logging.info("Max steps reached.")
                return


def predict_with_network(net, dataloader, device):
    outputs = []
    for batch in dataloader:
        device_batch = {
            k: v.to(device=device, non_blocking=True) for k, v in batch.items()
        }
        with torch.no_grad():
            mean, var = net(device_batch)
            mean, var = mean.detach().cpu().numpy(), var.detach().cpu().numpy()
            outputs.append(np.hstack([mean, var]))
    return np.concatenate(outputs, axis=0)


def predict(args, device, datasplits):
    """Predict with network."""
    # Setup model
    net = get_model(args)
    net.to(device)

    # Load model parameters
    model_path = os.path.join(args.output_dir, "best_model.pth")
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict["model"])

    # Setup data loaders
    loaders = {
        k: torch.utils.data.DataLoader(
            v, 32,
            collate_fn=data.CollateAtomsdata(pin_memory=device.type == "cuda")
        )
        for k, v in datasplits.items()
    }

    # Predict and save result
    logging.info("Start predicting...")
    for split_name, loader in loaders.items():
        predictions = predict_with_network(net, loader, device)
        output_path = os.path.join(args.output_dir, f"predictions_{split_name}.txt")
        np.savetxt(output_path, predictions)


if __name__ == "__main__":
    main()
