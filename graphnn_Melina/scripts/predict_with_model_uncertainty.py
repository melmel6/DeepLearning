import argparse
import json
import logging
import os

import numpy as np
import torch

from context import graphnn
from graphnn import data
from runner_uncertainty import get_model


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Run model pretrained model on dataset and obtain output",
        fromfile_prefix_chars="+",
    )
    parser.add_argument(
        "--model_dir", nargs="+", required=True,
        help="List of one or more model directories",
    )
    parser.add_argument(
        "--output_dir", type=str, default="", help="Predictions subdir.",
    )
    parser.add_argument(
        "--dataset", type=str, default="data/qm9.db", help="Path to ASE database",
    )
    parser.add_argument(
        "--target", type=str, default="U0", help="Name of target property",
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
        "--device",
        type=str,
        default="cuda",
        help="Set which device to use for training e.g. 'cuda' or 'cpu'",
    )
    return parser.parse_args(arg_list)


def predict_with_model(model, dataloader, device):
    outputs = []
    for batch in dataloader:
        device_batch = {
            k: v.to(device=device, non_blocking=True) for k, v in batch.items()
        }
        with torch.no_grad():
            mean, var = model(device_batch)
            mean, var = mean.detach().cpu().numpy(), var.detach().cpu().numpy()
            outputs.append(np.hstack([mean, var]))
    return np.concatenate(outputs, axis=0)


def main():
    args = get_arguments()

    # Setup output sub dirs
    output_dirs = [
        os.path.join(model_dir, args.output_dir) for model_dir in args.model_dir
    ]
    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[logging.FileHandler(
            os.path.join(output_dir, "predictions_log.txt"), mode="w"
        ) for output_dir in output_dirs] + [logging.StreamHandler()],
    )

    # Create device
    device = torch.device(args.device)
    # Put a tensor on the device before loading data
    # This way the GPU appears to be in use when other users run gpustat
    torch.tensor([0], device=device)

    # Load dataset
    logging.info("loading data %s", args.dataset)
    dataset = data.AseDbData(
        args.dataset,
        data.TransformRowToGraph(cutoff=args.cutoff, targets=args.target),
    )
    dataset = data.BufferData(dataset)  # Load data into host memory

    # predict with each model
    for i, model_dir in enumerate(args.model_dir):

        # Load runner arguments from model dir
        arg_path = os.path.join(model_dir, "arguments.json")
        with open(arg_path, "r") as arg_file:
            arg_dict = json.load(arg_file)
        runner_args = argparse.Namespace(**arg_dict)

        # Setup model
        net = get_model(runner_args)
        net.to(device)

        # Load model parameters
        model_path = os.path.join(model_dir, "best_model.pth")
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict["model"])

        # Load splits
        split_file_path = args.split_file or os.path.join(model_dir, "datasplits.json")
        with open(split_file_path, "r") as split_file:
            splits = json.load(split_file)

        # Split the dataset
        datasplits = {}
        for key, indices in splits.items():
            datasplits[key] = torch.utils.data.Subset(dataset, indices)

        # Setup data loaders
        loaders = {
            k: torch.utils.data.DataLoader(
                v, 32,
                collate_fn=data.CollateAtomsdata(pin_memory=device.type == "cuda")
            )
            for k, v in datasplits.items()
        }

        # Run model and save result
        logging.info(f"Predict with model: {model_dir}")
        for split_name, loader in loaders.items():
            predictions = predict_with_model(net, loader, device)
            output_path = os.path.join(output_dirs[i], f"predictions_{split_name}.txt")
            np.savetxt(output_path, predictions)
    logging.info("Done.")


if __name__ == "__main__":
    main()
