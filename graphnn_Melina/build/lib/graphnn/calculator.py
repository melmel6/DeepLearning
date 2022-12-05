from ase.calculators.calculator import Calculator, all_changes
from graphnn.data import TransformRowToGraphXyz, collate_atomsdata


class DummyRow:
    def __init__(self, atoms):
        self.atoms = atoms.copy()

    def toatoms(self):
        return self.atoms


class MLCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model,
        cutoff,
        energy_scale=1.0,
        forces_scale=1.0,
        stress_scale=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model = model
        self.model_device = next(model.parameters()).device
        self.cutoff = cutoff
        self.energy_scale = energy_scale
        self.forces_scale = forces_scale
        self.stress_scale = stress_scale

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)

        if self.calculation_required(atoms, properties):
            Calculator.calculate(self, atoms)

            transformer = TransformRowToGraphXyz(cutoff=self.cutoff)
            model_inputs = transformer(DummyRow(atoms))
            model_inputs = collate_atomsdata([model_inputs])
            model_inputs = {
                k: v.to(self.model_device) for (k, v) in model_inputs.items()
            }

            model_results = self.model(model_inputs)

            results = {}

            # Convert outputs to calculator format
            results["forces"] = (
                model_results["forces"][0].detach().cpu().numpy() * self.forces_scale
            )
            results["energy"] = (
                model_results["energy"][0].detach().cpu().numpy().item()
                * self.energy_scale
            )
            results["stress"] = (
                model_results["stress"][0].detach().cpu().numpy() * self.stress_scale
            )

            self.results = results
