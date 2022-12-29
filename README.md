# DeepLearning Project: Deep Evidential Regression on PAINN Graph Neural Network

### Alexandra Polymenopoulou - s212558
### Melina Siskou - s213158
### Paolo Federico - s212975

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

    $ python scripts/runner.py --help

Run:

    $ python scripts/runner.py --dataset data/qm9.db --max_steps 700000 --output_dir runs/final --device cuda --target G_ --learning_rate 0.000001


## Prediction

Once the model has been trained, we can use it to predict uncertainty

Run the `predict_with_model.py`. Specify the model directory with `--model_dir` and the dataset with `--dataset`

Run:



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
