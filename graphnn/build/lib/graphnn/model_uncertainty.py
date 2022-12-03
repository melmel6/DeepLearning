import math

import torch
from torch import nn

from graphnn import layer


class SchnetModel(nn.Module):
    """SchNet model with optional edge updates.

    Output: loc(x), scale(x)
    """

    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        update_edges=False,
        target_mean=[0.0],
        target_stddev=[1.0],
        normalize_atomwise=True,
        scale_transform="softplus",
        scale_eps=1e-6,
        **kwargs,
    ):
        """
        Args:
            num_interactions (int): Number of interaction layers
            hidden_state_size (int): Size of hidden node states
            cutoff (float): Atomic interaction cutoff distance [Å]
            update_edges (bool): Enable edge updates
            target_mean ([float]): Target normalisation constant
            target_stddev ([float]): Target normalisation constant
            normalize_atomwise (bool): Use atomwise normalisation
            scale_transform (str): Function to transform scale output
            scale_eps (float): Small number added to the scale output
        """
        super().__init__(**kwargs)
        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.gaussian_expansion_step = 0.1
        self.scale_eps = scale_eps

        # Setup function used to transform the scale output
        assert scale_transform in ["softplus", "exp"]
        if scale_transform == "softplus":
            self.scale_transform = torch.nn.functional.softplus
        if scale_transform == "exp":
            self.scale_transform = torch.exp

        num_embeddings = 119  # atomic numbers + 1
        edge_size = int(math.ceil(self.cutoff / self.gaussian_expansion_step))

        # Setup atom embeddings
        self.atom_embeddings = nn.Embedding(num_embeddings, hidden_state_size)

        # Setup interaction networks
        self.interactions = nn.ModuleList(
            [
                layer.Interaction(hidden_state_size, edge_size)
                for _ in range(num_interactions)
            ]
        )

        if update_edges:
            self.edge_updates = nn.ModuleList(
                [
                    layer.EdgeUpdate(edge_size, hidden_state_size)
                    for _ in range(num_interactions)
                ]
            )
        else:
            self.edge_updates = [lambda e_state, e, n: e_state] * num_interactions

        # Setup readout function
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            layer.ShiftedSoftplus(),
            nn.Linear(hidden_state_size, 2),
        )

        # Initialize bias of uncertainty output to a positive value
        self.readout_mlp[-1].bias.data[1] = 1.0

        # Normalisation constants
        self.normalize_atomwise = torch.nn.Parameter(
            torch.tensor(normalize_atomwise), requires_grad=False
        )
        self.normalize_loc = torch.nn.Parameter(
            torch.as_tensor(target_mean), requires_grad=False
        )
        self.normalize_scale = torch.nn.Parameter(
            torch.as_tensor(target_stddev), requires_grad=False
        )

    def forward(self, input_dict):
        """
        Args:
            input_dict (dict): Input dictionary of tensors with keys: nodes,
                               num_nodes, edges, edges_features, num_edges,
                               targets
        """
        # Unpad and concatenate edges and features into batch (0th) dimension
        edges_features = layer.unpad_and_cat(
            input_dict["edges_features"], input_dict["num_edges"]
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
        nodes = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])
        nodes = self.atom_embeddings(nodes)

        # Expand edge features in Gaussian basis
        edge_state = layer.gaussian_expansion(
            edges_features, [(0.0, self.gaussian_expansion_step, self.cutoff)]
        )

        # Apply interaction layers
        for edge_layer, int_layer in zip(self.edge_updates, self.interactions):
            edge_state = edge_layer(edge_state, edges, nodes)
            nodes = int_layer(nodes, edges, edge_state)

        # Apply readout function
        nodes = self.readout_mlp(nodes)

        # Obtain graph level output
        graph_output = layer.sum_splits(nodes, input_dict["num_nodes"])

        # Split loc and scale
        loc, scale = torch.split(graph_output, 1, dim=1)

        # Transform scale output
        scale = self.scale_transform(scale) + self.scale_eps

        # Apply denormalization
        loc = loc * self.normalize_scale
        if self.normalize_atomwise:
            loc = loc + self.normalize_loc * input_dict["num_nodes"].unsqueeze(1)
        else:
            loc = loc + self.normalize_loc

        return loc, scale
