"""Datasets: Toy graph datasets."""

import torch
import math


class GraphDataset():
    """Parent class for graph datasets.

    Provides some convenient properties and functions.

    To create a dataset, inherit from this class and specify the following
    member variables

    Member variables
    ----------------
    num_graphs        : Number of graphs
    node_coordinates  : 2-d coordinates of all nodes in all graphs
                        (shape num_nodes x 2)
    node_graph_index  : Graph index (between 0 and num_graphs-1) for each node
                        (shape num_nodes)
    edge_list         : Array of edges (shape num_edges x 2)
    """

    def __init__(self):
        self.num_graphs = 0
        self.node_coordinates = torch.empty((0, 2))
        self.node_graph_index = torch.empty((0))
        self.edge_list = torch.empty((0, 2))
        self._current_angle = 0.

    @property
    def node_from(self):
        """List of first nodes for each edge = edge_list[:, 0]."""
        return self.edge_list[:, 0]

    @property
    def node_to(self):
        """: List of first nodes for each edge = edge_list[:, 0]."""
        return self.edge_list[:, 1]

    @property
    def edge_graph_index(self):
        """Graph index for each edge."""
        return self.node_graph_index[self.node_to]

    @property
    def num_nodes(self):
        """Number of nodes."""
        return self.node_coordinates.shape[0]

    @property
    def num_edges(self):
        """Number of edges."""
        return self.edge_list.shape[0]

    @property
    def edge_vector_diffs(self):
        """Vector between nodes for each edge."""
        return (self.node_coordinates[self.node_to] -
                self.node_coordinates[self.node_from])

    @property
    def edge_lengths(self):
        """Length of each edge."""
        return self.edge_vector_diffs.norm(dim=1, keepdim=True)

    @property
    def edge_vectors(self):
        """Normalized vector between nodes for each edge."""
        return self.edge_vector_diffs / self.edge_lengths

    def center(self):
        """Centers each graph in the dataset on its mean coordinate.

        Returns
        -------
        GraphDataset
            Returns self to enable chaining operations

        """
        for i in range(self.num_graphs):
            coords = self.node_coordinates[self.node_graph_index == i]
            mean = coords.mean(0)
            self.node_coordinates[self.node_graph_index == i] = coords-mean
        return self

    def rotate(self, angle):
        """Rotates each graph in the data set individually around it's center.

        Parameters
        ----------
        angle : float
            Rotation angle (in radians)

        Returns
        -------
        GraphDataset
            Returns self to enable chaining operations

        """
        relative_angle = angle - self._current_angle
        self._current_angle = angle
        phi = torch.tensor(relative_angle)
        R = torch.tensor([[torch.cos(phi), -torch.sin(phi)],
                         [torch.sin(phi), torch.cos(phi)]])
        for i in range(self.num_graphs):
            coords = self.node_coordinates[self.node_graph_index == i]
            mean = coords.mean(0)
            self.node_coordinates[self.node_graph_index == i] = (
                coords-mean) @ R + mean
        return self


class Tetris(GraphDataset):
    """Tetris dataset.

    Contains 7 graphs corresponding to the shapes in Tetris. Each graph is
    fully connected, and the nodes have associated  2-d coordinates.
    """

    def __init__(self):
        super().__init__()

        # Graph list
        self.graph_list = torch.tensor([0, 1, 2, 3, 4, 5, 6])

        # Number of graphs in the dataset
        self.num_graphs = 7

        # Node coordinates
        self.node_coordinates = torch.tensor([
            [0, 0], [0, 1], [0, 2], [0, 3],
            [0, 0], [0, 1], [1, 1], [1, 0],
            [0, 0], [0, 1], [1, 1], [1, 2],
            [1, 0], [1, 1], [0, 1], [0, 2],
            [0, 0], [1, 0], [2, 0], [1, 1],
            [1, 0], [1, 1], [1, 2], [0, 2],
            [0, 0], [0, 1], [0, 2], [1, 2]
        ], dtype=torch.float)

        # Node graph index
        self.node_graph_index = torch.tensor([
            0, 0, 0, 0,
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,
            4, 4, 4, 4,
            5, 5, 5, 5,
            6, 6, 6, 6])

        # Edge list - generate fully conected graphs
        edges = []
        for i in range(7):
            for m in range(4):
                for n in range(4):
                    if n != m:
                        edges.append([m + i*4, n + i*4])
        self.edge_list = torch.tensor(edges)


class Circles(GraphDataset):
    """Circles dataset.

    Contains 6 graphs with 3-8 nodes placed in a circle 2-d coordinates.
    """

    def __init__(self):
        super().__init__()

        # Graph list
        self.graph_list = torch.tensor([0, 1, 2, 3, 4, 5])

        # Number of graphs in the dataset
        self.num_graphs = 6

        # Generate circle-graphs of varying size
        node_coordinates = []
        node_graph_index = []
        edges = []
        node_num = 0
        for i in range(6):
            for j in range(i+3):
                n = i+3
                a = 1 / math.sin(math.pi/n)/2
                x = j*math.pi*2/n
                node_coordinates.append([a*math.cos(x), a*math.sin(x)])
                node_graph_index.append(i)
                if j == 0:
                    edges.append([node_num, node_num+n-1])
                    edges.append([node_num+n-1, node_num])
                else:
                    edges.append([node_num, node_num-1])
                    edges.append([node_num-1, node_num])
                node_num += 1
        self.node_coordinates = torch.tensor(node_coordinates)
        self.node_graph_index = torch.tensor(node_graph_index)
        self.edge_list = torch.tensor(edges)
