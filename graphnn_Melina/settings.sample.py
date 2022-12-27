"""Developer settings

Copy these settings to a new file called 'settings.py'.
"""

# Settings for running scripts on the CogSys cluster

# DTU username
USER = "my-dtu-username"

# List of hostnames
HOSTS = """
mnemosyne.compute.dtu.dk
theia.compute.dtu.dk
phoebe.compute.dtu.dk
themis.compute.dtu.dk
oceanus.compute.dtu.dk
hyperion.compute.dtu.dk
coeus.compute.dtu.dk
cronus.compute.dtu.dk
crius.compute.dtu.dk
iapetus.compute.dtu.dk
""".strip().split()

# Path to git repository on remote
REPO_PATH = "./graphnn/"

# The conda envronment that will be activated on the remote
CONDA_ENV = "graphnn"


# Settings for running jobs on Niflheim

NIFLHEIM_LOGIN_HOST = "svol.fysik.dtu.dk"
NIFLHEIM_VIRTUAL_ENV = "~/graphnn_env"  # must be an absolute path
NIFLHEIM_REPO_SCRATCH = "~/graphnn_revisions"  # must be an absolute path
NIFLHEIM_PYTHON_MODULE = "Python/3.8.6-GCCcore-10.2.0"

# Adjust the preamble depending on computation needs
NIFLHEIM_SCRIPT_PREAMBLE = f"""#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=sm3090
#SBATCH -N 1      # Minimum of 1 node
#SBATCH -n 8     # 8 MPI processes per node
#SBATCH --time=2-00:00:00
#SBATCH --mem=10G     # 10 GB RAM per node
#SBATCH --gres=gpu:RTX3090:1  # Allocate 1 GPU
module load {NIFLHEIM_PYTHON_MODULE}
module load foss
source {NIFLHEIM_VIRTUAL_ENV}/bin/activate
"""
