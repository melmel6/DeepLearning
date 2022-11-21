"""Demonstrates molecular dynamics with constant temperature using ML model"""

import sys
import torch

from ase.lattice.cubic import FaceCenteredCubic
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase import units
import ase.db

from asap3 import EMT  # Way too slow with ase.EMT !

from context import graphnn
from graphnn.calculator import MLCalculator
from graphnn.model_forces import SchnetModelForces


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = 500  # Kelvin, simulation temperature

# Load molecule from MD17 dataset
with ase.db.connect("../data/md17aspirin.db") as db:
    row = next(db.select(1))
    atoms = row.toatoms()
    atoms.center(vacuum=2)

cutoff = 5

# Load ML model
model = SchnetModelForces(num_interactions=6, hidden_state_size=128, cutoff=cutoff,)
model_path = "../data/md17_aspirin_best_model.pth"
state_dict = torch.load(model_path)
model.load_state_dict(state_dict["model"])
model.to(device)

# ML model was trained with kcal/mol as energy, but ASE uses eV
scale = units.kcal / units.mol
mlcalc = MLCalculator(
    model=model, cutoff=cutoff, energy_scale=scale, forces_scale=scale
)

# Set calculator
atoms.calc = mlcalc

# We want to run MD with constant energy using the Langevin algorithm
# with a time step of 1 fs, the temperature T and the friction
# coefficient to 0.02 atomic units.
dyn = Langevin(atoms, 1 * units.fs, T * units.kB, 0.002)


def printenergy(a=atoms):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print(
        "Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  "
        "Etot = %.3feV" % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin)
    )


dyn.attach(printenergy, interval=50)

# We also want to save the positions of all atoms after every 100th time step.
traj = Trajectory("moldyn.traj", "w", atoms)
dyn.attach(traj.write, interval=1)

# Now run the dynamics
printenergy()
dyn.run(50000)
