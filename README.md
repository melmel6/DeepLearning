# DeepLearning

# GraphNN

Graph Neural Network model.

## Setup

Create and activate a Python virtual environment and then install the requirements specified in `requirements.txt`.
Note that the `asap3` package needs to be installed with `pip`. The other dependencies are available with `conda`.

The core functionality can (optionally) be installed as a Python package for use in other projects:

    $ python setup.py install

Note: Installing the Python package is not required for running the scripts in `/scripts`

## Data

Large files, such as datasets, are managed with [DVC]([https://dvc.org/]) and [Git](https://git-scm.com/).
If you are not familiar with DVC, take a look at the [docs](https://dvc.org/doc).

### DVC setup

See the [docs](https://dvc.org/doc) for how to install DVC either globally or locally in a virtual environment.
You need to install the ssh dependencies to use the remote repository.

The remote DVC repository is located on [Niflheim](https://wiki.fysik.dtu.dk/niflheim/) and you need a user account and to be on the DTU network to access it.

If your local username is different from your Niflheim username, you may have to set the username locally in the graphnn folder:

    $ cd graphnn/
    $ cp .dvc/config .dvc/config.local
    $ dvc remote modify --local niflheim user [username]

### Pull data from remote repository

    $ dvc pull data/filename.csv.dvc

### Add data to version control and push to remote repository

    $ dvc add data/filename.csv
    $ git add data/.gitignore data/filename.csv.dvc
    $ git commit -m "dvc add data/filename.csv"
    $ dvc push

If you want to go more in depth with DVC, realpython has a thorough, easy-to-follow [tutorial](https://realpython.com/python-data-version-control/) that also touches on how to use DVC to organise and ensure reproducible experiments.

## Training the model with energy and forces

The main training script is `runner.py`.

Inspect `runner.py` arguments:

    $ python scripts/runner.py --help

Run:

    $ python scripts/runner.py [optional arguments]

To train with forces use the `runner_forces.py` script:

    $ python runner_forces.py --help

Some recommended settings are `--atomwise_normalization`, `--forces_weight 0.99`, `--update_edges`
and also use the `--dataset [path]` argument and point to an ASE database file with the dataset.

### Continue training

It is possible to continue running from a saved checkpoint file.
Use the `--load_model` argument and point to the saved model file.
It is saved as `best_model.pth` in the output directory.
The other command-line arguments should be the same as in the previous run
or you can load the parameters from a text file with the `+` prefix, e.g.

    $ python runner_forces.py +runs/model_output/commandline_args.txt

When reloading a model to continue training it is important to specify
the `--split_file` argument. Otherwise the training will restart with a new random train/validation split,
which will be different from the one originally used for training.
The `datasplits.json` in the output directory contains the splits used.

## Using trained model as a calculator

Once the model has been trained, we can use it to predict properties of new atomic systems

Run the `predict_with_model.py` or `predict_with_model_forces.py` (if you have trained with forces).
Specify the model directory with `--model_dir` and the dataset with `--dataset`

It is also possible to wrap the model in an ASE calculator to run molecular dynamics.
See `scripts/md_example.py` for an example using a pretrained model on an aspirin dataset.

## Fabric

Some common tasks can be executed with [Fabric](http://www.fabfile.org/), such as training a model on a remote server.
There is currently commands for running tasks on the [CogSys GPU Cluster](https://itswiki.compute.dtu.dk/index.php/GPU_Cluster) and on [Niflheim](https://wiki.fysik.dtu.dk/niflheim/Niflheim7_Getting_started).

### Setup

Setup SSH keys etc. to access the remote server, clone this repository and download any data you need to the server.

Note: You need to be connected to the DTU network locally or via VPN to access DTU servers.

On your local machine, copy `settings.sample.py` to a file named `settings.py` and fill in the details.

Install Fabric with pip:

    $ pip install fabric

List available commands:

    $ fab --list

Below are a few examples.
Inspect `fabfile.py` for more details.

### CogSys GPU CLuster

The gpustat command checks all GPUs on the various servers at DTU Compute. Run it to check if your fabric can communicate with the servers.

    $ fab gpustat

Run script on remote host example:

    $ fab run themis 0 "scripts/runner.py --target=U0"

### Niflheim

Submit job:

    $ fab submit "scripts/runner.py --dataset=/abosulute/path/to/dataset"

List running jobs:

    $ fab qstat

## Test

Run all tests:

    $ python -m unittest




# Evidential Deep Learning

<h3 align='center'>"All models are wrong, but some — <i>that know when they can be trusted</i> — are useful!"</h3>
<p align='right'><i>- George Box (Adapted)</i></p>


![](assets/banner.png)

This repository contains the code to reproduce [Deep Evidential Regression](https://proceedings.neurips.cc/paper/2020/file/aab085461de182608ee9f607f3f7d18f-Paper.pdf), as published in [NeurIPS 2020](https://neurips.cc/), as well as more general code to leverage evidential learning to train neural networks to learn their own measures of uncertainty directly from data!

## Setup
To use this package, you must install the following dependencies first: 
- python (>=3.7)
- tensorflow (>=2.0)
- pytorch (support coming soon)

Now you can install to start adding evidential layers and losses to your models!
```
pip install evidential-deep-learning
```
Now you're ready to start using this package directly as part of your existing `tf.keras` model pipelines (`Sequential`, `Functional`, or `model-subclassing`):
```
>>> import evidential_deep_learning as edl
```

### Example
To use evidential deep learning, you must edit the last layer of your model to be *evidential* and use a supported loss function to train the system end-to-end. This repository supports evidential layers for both fully connected and convolutional (2D) layers. The evidential prior distribution presented in the paper follows a Normal Inverse-Gamma and can be added to your model: 

```
import evidential_deep_learning as edl
import tensorflow as tf

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        edl.layers.DenseNormalGamma(1), # Evidential distribution!
    ]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3), 
    loss=edl.losses.EvidentialRegression # Evidential loss!
)
```

![](assets/animation.gif)
Checkout `hello_world.py` for an end-to-end toy example walking through this step-by-step. For more complex examples, scaling up to computer vision problems (where we learn to predict tens of thousands of evidential distributions simultaneously!), please refer to the NeurIPS 2020 paper, and the reproducibility section of this repo to run those examples. 

## Reproducibility
All of the results published as part of our NeurIPS paper can be reproduced as part of this repository. Please refer to [the reproducibility section](./neurips2020) for details and instructions to obtain each result. 

## Citation
If you use this code for evidential learning as part of your project or paper, please cite the following work:  

    @article{amini2020deep,
      title={Deep evidential regression},
      author={Amini, Alexander and Schwarting, Wilko and Soleimany, Ava and Rus, Daniela},
      journal={Advances in Neural Information Processing Systems},
      volume={33},
      year={2020}
    }
