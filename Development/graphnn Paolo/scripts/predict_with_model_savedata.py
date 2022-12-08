import argparse
import torch
import logging
import os
import json
import runner
import numpy as np

from context import graphnn
from graphnn import data


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Run model pretrained model on dataset and obtain output",
        fromfile_prefix_chars="+",
    )
    parser.add_argument(
        "--model_dir", type=str, default=None, help="Directory of model",
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Path to output directory",
    )
    parser.add_argument(
        "--dataset", type=str, default="data/qm9.db", help="Path to ASE database",
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
            outputs.append(model(device_batch).detach().cpu().numpy())

    return np.concatenate(outputs, axis=0)


def main():
    args = get_arguments()

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "predict_with_model_log.txt"), mode="w"
            ),
            logging.StreamHandler(),
        ],
    )

    # Load runner arguments from output dir
    arg_path = os.path.join(args.model_dir, "arguments.json")
    with open(arg_path, "r") as arg_file:
        arg_dict = json.load(arg_file)
    runner_args = argparse.Namespace(**arg_dict)

    # Setup model
    device = torch.device(args.device)
    net = runner.get_model(runner_args)
    net.to(device)

    # Load model parameters
    model_path = os.path.join(args.model_dir, "best_model.pth")
    state_dict = torch.load(model_path)
    net.load_state_dict(state_dict["model"])

    # Setup dataset
    logging.info("loading data %s", args.dataset)
    dataset = data.AseDbData(
        args.dataset,
        data.TransformRowToGraph(cutoff=runner_args.cutoff, targets=runner_args.target),
    )
    dataset = data.BufferData(dataset)  # Load data into host memory

    # Load splits
    if args.split_file:
        with open(args.split_file, "r") as fp:
            splits = json.load(fp)

    # Split the dataset
    datasplits = {}
    for key, indices in splits.items():
        datasplits[key] = torch.utils.data.Subset(dataset, indices)

    # Setup loaders
    loaders = {
        k: torch.utils.data.DataLoader(v, 32, collate_fn=data.collate_atomsdata)
        for k, v in datasplits.items()
    }

    # Run model and save result
    for splitname, loader in loaders.items():
        predictions = predict_with_model(net, loader, device)
        output_path = os.path.join(args.output_dir, "predictions_%s.txt" % splitname)
        np.savetxt(output_path, predictions)
        with open(output_path, 'w') as fp:
            json.dump(predictions, fp)
        output_path = os.path.join(args.output_dir, "data_%s.txt" % splitname)
        with open(output_path, 'w') as fp:
            json.dump(loader, fp)


if __name__ == "__main__":
    main()
