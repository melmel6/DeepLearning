import argparse
import itertools
import json
import logging
import math
import os
import sys

import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from context import graphnn
from graphnn import data, model, continuous_loss, continuous_loss_pytorch

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
        "--node_size", type=int, default=64, help="Size of hidden node states"  # Hidden_state_size
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/model_output",
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
        "--target", type=str, default="energy", help="Name of target property",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.000001, help="Learning rate",
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


def eval_model(model, dataloader, device):
    running_ae = 0
    running_se = 0
    running_count = 0
    for batch in dataloader:
        device_batch = {
            k: v.to(device=device, non_blocking=True) for k, v in batch.items()
        }
        with torch.no_grad():
            outputs = torch.index_select(input=model(device_batch)[0].detach().cpu(), dim=1, index=torch.tensor([0]))
            outputs = outputs.numpy()
        targets = batch["targets"].detach().cpu().numpy()
        #print('outputs')
        #print(outputs)
        #print('targets')
        #print(targets)
        running_ae += np.sum(np.abs(targets - outputs), axis=0)
        running_se += np.sum(np.square(targets - outputs), axis=0)
        running_count += targets.shape[0]
        #print('running_ae')
        #print(running_ae)
        #print('running_se')
        #print(running_se)
        #print('running_count')
        #print(running_count)
    mae = running_ae / running_count
    #print('mae')
    #print(mae)
    rmse = np.sqrt(running_se / running_count)
    #print('rmse')
    #print(rmse)

    return mae, rmse


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
    net = model.SchnetModel(
        num_interactions=args.num_interactions,
        hidden_state_size=args.node_size,
        cutoff=args.cutoff,
        update_edges=args.update_edges,
        normalize_atomwise=args.atomwise_normalization,
        **kwargs
    )
    return net

# Custom loss function to handle the custom regularizer coefficient
def EvidentialRegressionLoss(true, pred):
    return continuous_loss_pytorch.EvidentialRegression(true, pred, lmbda=1e-4)
    # return continuous_loss.EvidentialRegression(true, pred, coeff=1e-2)

# =============================================================================
# def plot_epistemic(dictionary, aleatoric, epistemic, x='num_nodes', y='H', cmap='cool'):
#     X = dictionary[x]
#     Y = dictionary[y]
#     print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
#     print(aleatoric, epistemic)
#     return plt.scatter(X, Y, s=200, c=epistemic, cmap=cmap)
#
# =============================================================================
    

def main():
    args = get_arguments()

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "#printlog.txt"), mode="w"
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
        args.dataset, data.TransformRowToGraph(cutoff=args.cutoff, targets=args.target)
    )

    dataset = data.BufferData(dataset)  # Load data into host memory

    # print(dataset.data_objects)

    maxNode = max(dataset.data_objects, key=lambda x:x['num_nodes'])
    minNode = min(dataset.data_objects, key=lambda x:x['num_nodes'])

    print(maxNode['num_nodes'])
    print(minNode['num_nodes'])

    # maxH = max(dataset.data_objects, key=lambda x:x['H_'])
    # minH = min(dataset.data_objects, key=lambda x:x['H_'])

    # print(maxH['H_'])
    # print(minH['H_'])

    # dataset.data_objects = [d for d in dataset.data_objects if (d['num_nodes']>15 and d['H_']<-72)]

    # Subsetting:
    # dataset.data_objects = [d for d in dataset.data_objects if (d['num_nodes']>15 and d['H_']<-74)]
    # dataset.data_objects = [d for d in dataset.data_objects if (d['num_nodes']>11)]


    # print(dataset.data_objects)
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
        100,
        sampler=torch.utils.data.RandomSampler(datasplits["train"]),
        collate_fn=data.CollateAtomsdata(pin_memory=device.type == "cuda")
    )
    val_loader = torch.utils.data.DataLoader(
        datasplits["validation"],
        32,
        collate_fn=data.CollateAtomsdata(pin_memory=device.type == "cuda")
    )

    # Initialise model
    net = get_model(args, target_mean=target_mean, target_stddev=target_stddev)
    net = net.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    # criterion = torch.nn.MSELoss()
    scheduler_fn = lambda step: 0.96 ** (step / 100000)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_fn)

    log_interval = 10000
    running_loss = 0
    running_loss_count = 0
    best_val_mae = np.inf
    step = 0
    # Restore checkpoint - Load model from previous run
    if args.load_model:
        state_dict = torch.load(args.load_model)
        net.load_state_dict(state_dict["model"])
        step = state_dict["step"]
        best_val_mae = state_dict["best_val_mae"]
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
            outputs, aleatoric_uncertainty, epistemic_uncertainty = net(batch)
            #print('outputs')
            #print(outputs.shape)
            #print(outputs)
            # print("*loss*")
            loss = EvidentialRegressionLoss(batch["targets"], outputs)
            # print("*end loss*")
            #print('loss')
            #print(loss.shape)
            #print(loss)
            loss.backward()   # Added gradient because loss is tensor, not scalar
            optimizer.step()

            loss_value = loss.item()
            running_loss += loss_value * batch["targets"].shape[0]
            running_loss_count += batch["targets"].shape[0]

            # #print(step, loss_value)
            # Validate and save model
            if (step % log_interval == 0) or ((step + 1) == args.max_steps):
                train_loss = running_loss / running_loss_count
                running_loss = running_loss_count = 0

                val_mae, val_rmse = eval_model(net, val_loader, device)

                logging.info(
                    "step=%d, val_mae=%g, val_rmse=%g, train_loss=%g",
                    step,
                    val_mae,
                    val_rmse,
                    train_loss,
                )

                # Save checkpoint
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    torch.save(
                        {
                            "model": net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "step": step,
                            "best_val_mae": best_val_mae,
                        },
                        os.path.join(args.output_dir, "best_model.pth"),
                    )
            step += 1

            scheduler.step()

            if step >= args.max_steps:
                logging.info("Max steps reached, exiting")
                sys.exit(0)

        # =============================================================================
# =============================================================================
# #         # Plot uncertainty
# #         print('_________UUUUUUNNNNNNNCCCCERRRTAINTYYYYYYYYYYYYYYYYYYYYY______________')
# #         X = 'num_nodes'
# #         Y = 'num_edges'
# #        
# #         # Setup the normalization and the colormap
# #         normalize = mcolors.Normalize(vmin=epistemic_uncertainty.detach().min(), vmax=epistemic_uncertainty.detach().max())
# #         colormap = cm.cool
# #        
# #         # Plot
# #         plt.scatter(batch[X], batch[Y], s=150, c=epistemic_uncertainty.detach(), cmap='cool', alpha=0.4)
# #
# #         # Setup the colorbar
# #         scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
# #         scalarmappaple.set_array(epistemic_uncertainty.detach())
# #         plt.colorbar(scalarmappaple)
# #        
# #         # Title and labels
# #         plt.title('Epistemic uncertainty')
# #         plt.xlabel(X)
# #         plt.ylabel(Y)
# #        
# #         # Save plot
# #         plt.savefig('uncertainty Plots\Epistemic_' + str(epoch) + '.svg', bbox_inches='tight')    
# #         plt.savefig('uncertainty Plots\Epistemic_' + str(epoch) + '.pdf', bbox_inches='tight')  
# #         plt.savefig('uncertainty Plots\Epistemic_' + str(epoch) + '.png', bbox_inches='tight', dpi=96)  
# #         plt.close()
# =============================================================================
# =============================================================================
    

if __name__ == "__main__":
    main()
