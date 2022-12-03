import unittest
import sys
import os
import torch

sys.path.append("..")

from graphnn.data import (
    AseDbData,
    TransformRowToGraph,
    collate_atomsdata,
    TransformRowToGraphXyz,
)


def setUpModule():
    import ase
    import ase.build

    with ase.db.connect("test.db", append=False) as db:
        for molecule, energy in zip(["H2O", "CS", "HCl"], [1.0, 2.0, 3.0]):
            atoms = ase.build.molecule(molecule)
            db.write(atoms, key_value_pairs={"lumo": energy})

    with ase.db.connect("test_solids.db", append=False) as db:
        atoms = ase.Atoms(
            ["C", "H"],
            cell=[[1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1.0]],
            scaled_positions=[[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]],
            pbc=True,
        )
        db.write(atoms, key_value_pairs={"lumo": 0.2})
        atoms = ase.Atoms(
            ["C", "H", "H"],
            cell=[[1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1.0]],
            scaled_positions=[[0.5, 0.5, 0.5], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]],
            pbc=True,
        )
        db.write(atoms, key_value_pairs={"lumo": 0.3})


def tearDownModule():
    os.remove("test.db")
    os.remove("test_solids.db")


class TestLoader(unittest.TestCase):
    def test_data(self):
        transformer = TransformRowToGraph(10, targets="lumo")
        # Setup dataset
        dataset = AseDbData("test.db", transformer)
        # Extract one sample
        entry = dataset[0]

        self.assertTrue(all([8, 1, 1] == entry["nodes"].detach().numpy()))

    def test_loader_compatibility(self):
        transformer = TransformRowToGraph(10, targets="lumo")
        # Setup dataset
        dataset = AseDbData("test.db", transformer)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=2, collate_fn=collate_atomsdata
        )
        for i, batch in enumerate(dataloader):
            if i == 0:
                self.assertTrue(all([3, 2] == batch["num_nodes"].detach().numpy()))
            elif i == 1:
                self.assertTrue(all([2] == batch["num_nodes"].detach().numpy()))

    def test_iterator(self):
        transformer = TransformRowToGraph(10, targets="lumo")
        dataset = AseDbData("test.db", transformer)
        for i, sample in enumerate(dataset):
            pass


class TestSolidsLoader(unittest.TestCase):
    def test_displacement(self):
        from graphnn import layer
        import ase
        import numpy as np

        cutoff = 0.8
        transformer = TransformRowToGraphXyz(cutoff)
        dataset = AseDbData("test_solids.db", transformer)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=2, collate_fn=collate_atomsdata
        )

        with ase.db.connect("test_solids.db") as db:
            reference_edges = []
            reference_dists = []
            reference_displacements = []
            for row in db.select():
                dists = []
                edges = []
                displacements = []

                atoms = row.toatoms()
                neighborlist = ase.neighborlist.NewPrimitiveNeighborList(
                    cutoff, skin=0.0, self_interaction=False, bothways=True
                )
                neighborlist.build(
                    atoms.get_pbc(), atoms.get_cell(), atoms.get_positions()
                )
                for i in range(len(atoms)):
                    indices, offsets = neighborlist.get_neighbors(i)
                    rel_positions = (
                        atoms.get_positions()[indices]
                        + offsets @ atoms.get_cell()
                        - atoms.get_positions()[i][None]
                    )
                    dist = np.sqrt(np.sum(np.square(rel_positions), axis=1))
                    dists.append(dist)

                    self_index = np.ones_like(indices) * i
                    this_edges = np.stack((indices, self_index), axis=1)
                    edges.append(this_edges)
                    displacements.append(offsets)

                reference_edges.append(np.concatenate(edges))
                reference_dists.append(np.concatenate(dists))
                reference_displacements.append(np.concatenate(displacements))

        for input_dict in dataloader:
            # Unpad and concatenate edges and features into batch (0th) dimension
            edges_displacement = layer.unpad_and_cat(
                input_dict["edges_displacement"], input_dict["num_edges"]
            )
            edge_offset = torch.cumsum(
                torch.cat(
                    (
                        torch.tensor([0], device=input_dict["num_nodes"].device),
                        input_dict["num_nodes"][:-1],
                    )
                ),
                dim=0,
            )
            edge_offset = edge_offset[:, None, None]
            edges = input_dict["edges"] + edge_offset
            edges = layer.unpad_and_cat(edges, input_dict["num_edges"])

            # Unpad and concatenate all nodes into batch (0th) dimension
            nodes_xyz = layer.unpad_and_cat(
                input_dict["nodes_xyz"], input_dict["num_nodes"]
            )

            # Compute edge distances
            edges_distance = layer.calc_distance(
                nodes_xyz,
                input_dict["cell"],
                edges,
                edges_displacement,
                input_dict["num_edges"],
            )

            n_edges = list(input_dict["num_edges"].detach().numpy())
            list_edges_distance = torch.split(edges_distance, n_edges)
            list_edges_displacement = torch.split(edges_displacement, n_edges)
            list_edges = torch.split(edges, n_edges)

            for edges_dist, edges_disp, edges, ref_dist, ref_disp, ref_edges in zip(
                list_edges_distance,
                list_edges_displacement,
                list_edges,
                reference_dists,
                reference_displacements,
                reference_edges,
            ):
                reference_key = np.concatenate((ref_disp, ref_edges), axis=1)
                key = torch.cat((edges_disp, edges), axis=1).detach().numpy()
                sorted_ref_dists = ref_dist[np.lexsort(reference_key.T)]
                sorted_dists = edges_dist.detach().numpy().squeeze()[np.lexsort(key.T)]
                assert np.allclose(sorted_dists, sorted_ref_dists)


if __name__ == "__main__":
    unittest.main()
