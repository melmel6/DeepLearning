# Deep Evidential Regression on PAINN Graph Neural Network

##### Alexandra Polymenopoulou - s212558 , Melina Siskou - s213158, Paolo Federico - s212975

<h3 align='center'>"All models are wrong, but some — <i>that know when they can be trusted</i> — are useful!"</h3>
<p align='right'><i>- George Box (Adapted)</i></p>

## Setup

Create and activate a Python virtual environment and then install the requirements specified in `requirements.txt`.
Note that the `asap3` package needs to be installed with `pip`. The other dependencies are available with `conda`.

The core functionality can (optionally) be installed as a Python package for use in other projects:

    $ python setup.py install

Note: Installing the Python package is not required for running the scripts in `/scripts`

## Data

Dataset should be downloaded into data folder from Dropbox https://www.dropbox.com/s/oxa4pb7p1c3fecq/qm9.db?dl=0


## Training the model

The main training script is `runner.py`.

Inspect `runner.py` arguments:

    $ python3 scripts/runner.py --help

Run:

    $ python3 scripts/runner.py --dataset data/qm9.db --max_steps 700000 --output_dir runs/final --device cuda --target G_ --learning_rate 0.000001

To check results from our training in **HPC**, check folder runs/HPC_training
You could also check the jobscript used in scripts/jobscript.sh

## Prediction

Once the model has been trained, we can use it to predict uncertainty

Run the `predict_with_model.py`. Specify the model directory with `--model_dir` and the dataset with `--dataset`

Run:

    $ python3 scripts/predict_with_model.py --model_dir runs/final --output_dir runs/final_predictions --split_file runs/final/datasplits.json --device cpu

