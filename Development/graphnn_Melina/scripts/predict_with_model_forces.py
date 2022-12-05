import argparse
import logging
import os
import json
import numpy as np
import torch

import runner_forces
from context import graphnn


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Run model pretrained model on dataset and obtain output",
        fromfile_prefix_chars="+",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory of model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Path to output directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/qm9.db",
        help="Path to ASE database",
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

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate predictions against target values",
    )

    return parser.parse_args(arg_list)


def predict_with_model(model, dataloader, device, return_targets=False):
    energy_outputs = []
    forces_outputs = []
    target_energy = []
    target_forces = []
    for batch in dataloader:
        device_batch = {
            k: v.to(device=device, non_blocking=True) for k, v in batch.items()
        }
        res = model(device_batch)
        for row, num_nodes in zip(
            res["forces"].detach().cpu().numpy(), batch["num_nodes"].cpu().numpy()
        ):
            forces_outputs.append(row[0:num_nodes])
        energy_outputs.append(res["energy"].detach().cpu().numpy())
        if return_targets:
            target_energy.append(batch["energy"].detach().cpu().numpy())
            for row, num_nodes in zip(
                batch["forces"].detach().cpu().numpy(), batch["num_nodes"].cpu().numpy()
            ):
                target_forces.append(row[0:num_nodes])

    if return_targets:
        return (
            np.concatenate(energy_outputs, axis=0),
            forces_outputs,
            np.concatenate(target_energy, axis=0),
            target_forces,
        )
    else:
        return np.concatenate(energy_outputs, axis=0), forces_outputs


def eval_predictions(target_energy, predicted_energy, target_forces, predicted_forces):
    diff_energy = predicted_energy - target_energy

    energy_mae = np.mean(np.abs(diff_energy))
    energy_rmse = np.sqrt(np.mean(np.square(diff_energy)))

    forces_running_l2_ae = 0
    forces_running_l2_se = 0
    forces_running_c_ae = 0
    forces_running_c_se = 0
    forces_running_count = 0
    for f_target, f_predict in zip(target_forces, predicted_forces):
        f_diff = f_predict - f_target
        forces_l2_norm = np.sqrt(np.sum(np.square(f_diff), axis=1))
        forces_running_c_ae += np.sum(np.abs(f_diff))
        forces_running_c_se += np.sum(np.square(f_diff))
        forces_running_l2_ae += np.sum(np.abs(forces_l2_norm))
        forces_running_l2_se += np.sum(np.square(forces_l2_norm))
        forces_running_count += forces_l2_norm.shape[0]

    forces_l2_mae = forces_running_l2_ae / forces_running_count
    forces_l2_rmse = np.sqrt(forces_running_l2_se / forces_running_count)

    forces_c_mae = forces_running_c_ae / (forces_running_count * 3)
    forces_c_rmse = np.sqrt(forces_running_c_se / (forces_running_count * 3))

    evaluation = {
        "energy_mae": energy_mae,
        "energy_rmse": energy_rmse,
        "forces_l2_mae": forces_l2_mae,
        "forces_l2_rmse": forces_l2_rmse,
        "forces_mae": forces_c_mae,
        "forces_rmse": forces_c_rmse,
    }

    return evaluation


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
    net = runner_forces.get_model(runner_args)
    for param in net.parameters():
        param.requires_grad = False
    net.to(device)

    # Load model parameters
    model_path = os.path.join(args.model_dir, "best_model.pth")
    state_dict = torch.load(model_path)
    net.load_state_dict(state_dict["model"])

    # Setup dataset
    logging.info("loading data %s", args.dataset)
    dataset = graphnn.data.AseDbData(
        args.dataset,
        graphnn.data.TransformRowToGraphXyz(
            cutoff=runner_args.cutoff,
            energy_property=runner_args.energy_property,
            forces_property=runner_args.forces_property,
        ),
    )
    dataset = graphnn.data.BufferData(dataset)  # Load data into host memory

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
        k: torch.utils.data.DataLoader(v, 2, collate_fn=graphnn.data.collate_atomsdata)
        for k, v in datasplits.items()
    }

    # Run model and save result
    for splitname, loader in loaders.items():
        logging.info("Running model on split: %s", splitname)
        if args.evaluate:
            (
                energy_predictions,
                forces_predictions,
                energy_targets,
                forces_targets,
            ) = predict_with_model(net, loader, device, args.evaluate)
        else:
            energy_predictions, forces_predictions = predict_with_model(
                net, loader, device, args.evaluate
            )
        output_path_energy = os.path.join(
            args.output_dir, "predictions_energy_%s.txt" % splitname
        )
        output_path_forces = os.path.join(
            args.output_dir, "predictions_forces_%s.txt" % splitname
        )
        np.savetxt(output_path_energy, energy_predictions)
        with open(output_path_forces, "w") as f:
            for res in forces_predictions:
                atoms = ["%g,%g,%g" % (a[0], a[1], a[2]) for a in res]
                f.write(",".join(atoms))
                f.write("\n")

        if args.evaluate:
            eval_dict = eval_predictions(
                energy_targets, energy_predictions, forces_targets, forces_predictions
            )
            eval_formatted = ", ".join(
                ["split=%s" % splitname]
                + ["%s=%f" % (k, v) for (k, v) in eval_dict.items()]
            )
            logging.info(eval_formatted)

    logging.info("Done")


if __name__ == "__main__":
    main()
