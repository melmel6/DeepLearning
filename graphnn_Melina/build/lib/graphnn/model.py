import math

import torch
from torch import nn

from graphnn import layer, newev


class SchnetModel(nn.Module):
    """SchNet model with optional edge updates."""

    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        update_edges=False,
        target_mean=[0.0],
        target_stddev=[1.0],
        normalize_atomwise=True,
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
        """
        super().__init__(**kwargs)
        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.gaussian_expansion_step = 0.1

        num_embeddings = 119  # atomic numbers + 1
        edge_size = int(math.ceil(self.cutoff / self.gaussian_expansion_step))

        # Setup atom embeddings
        self.atom_embeddings = nn.Embedding(num_embeddings, hidden_state_size)

        # Setup interaction networks
        self.interactions = nn.ModuleList(
            [
                layer.Interaction(hidden_state_size, edge_size)
                for _ in range(num_interactions) # num_interaction = "Number of interaction layers used"
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
            nn.Linear(hidden_state_size, 1),
            # newev.DenseNormalGamma_torch(1) # Evidential layer
        )
        
        # Setup evidential layer
        self.evidential = newev.DenseNormalGamma_torch(1)
        

        # Normalisation constants
        self.normalize_atomwise = torch.nn.Parameter(
            torch.tensor(normalize_atomwise), requires_grad=False
        )
        self.normalize_stddev = torch.nn.Parameter(
            torch.as_tensor(target_stddev), requires_grad=False
        )
        self.normalize_mean = torch.nn.Parameter(
            torch.as_tensor(target_mean), requires_grad=False
        )

    def forward(self, input_dict):  
        """
        Args:
            input_dict (dict): Input dictionary of tensors with keys: nodes,
                               num_nodes, edges, edges_features, num_edges,
                               targets
        """
        # Unpad and concatenate edges and features into batch (0th) dimension
        #print('input_dict')
        #print(input_dict)
        edges_features = layer.unpad_and_cat(
            input_dict["edges_features"], input_dict["num_edges"]   # Removes the filling zeros
        )
        #print('edges_features')
        #print(edges_features.shape)
        #print(edges_features)
        edge_offset = torch.cumsum(     # Cumulative sum of the number of nodes
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0, # [0, 10, 18, ...., 120, 129]
        )
        #print('edge_offset')
        #print(edge_offset.shape)
        #print(edge_offset)
        edge_offset = edge_offset[:, None, None] # [[[0]], [[10]], [[18]]...] match edges shape
        #print(edge_offset.shape)
        #print(edge_offset)
        edges = input_dict["edges"] + edge_offset # Adding the cumsum, each node of the edges has a unique id in the whole dataset
        #print('edges')
        #print(edges.shape)
        #print(edges)
        edges = layer.unpad_and_cat(edges, input_dict["num_edges"])
        #print(edges.shape)
        #print(edges)

        # Unpad and concatenate all nodes into batch (0th) dimension
        nodes = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])
        #print('nodes')
        #print(nodes.shape)
        #print(nodes)
        nodes = self.atom_embeddings(nodes) #To each atomic number (i think) corresponds a list of values as long as hidden_state (default 64)
        #print(nodes.shape)
        #print(nodes)
        
        # Expand edge features in Gaussian basis
        edge_state = layer.gaussian_expansion(  # 1 value becomes a list of 50 values 
            edges_features, [(0.0, self.gaussian_expansion_step, self.cutoff)]
        )
        #print('edge_state')
        #print(edge_state.shape)
        #print(edge_state)

        # Apply interaction layers
        for edge_layer, int_layer in zip(self.edge_updates, self.interactions): # The self. here seem to be related to the model structure, we might need to change them
            edge_state = edge_layer(edge_state, edges, nodes)   # Similar to sequential
            nodes = int_layer(nodes, edges, edge_state)
        #print('edge_state after interaction')
        #print(edge_state.shape)
        #print(edge_state)
        #print('nodes after interaction')
        #print(nodes.shape)
        #print(nodes)
        # Apply readout function
        nodes = self.readout_mlp(nodes)
        
# =============================================================================
#         # Apply evidential layer (convert to tensorflow and back)
#         nodes = nodes.detach().numpy()
#         nodes = tf.convert_to_tensor(nodes)
#         
#         nodes = edl.layers.DenseNormalGamma(1)(nodes)
#         
#         nodes = nodes.numpy()
#         nodes = torch.from_numpy(nodes)
# =============================================================================
        #print('nodes after readout_mlp')
        #print(nodes.shape)
        #print(nodes)
        #PROBABLY ALL THAT FOLLOWS SHOLD BE APPLIED BEFORE THE EVIDENTIAL LAYER
        # Obtain graph level output
        graph_output = layer.sum_splits(nodes, input_dict["num_nodes"]) # Sum nodes values for each molecule
        #print('graph_ouptut')
        #print(graph_output.shape)
        #print(graph_output)
        # Apply (de-)normalization
        normalizer = self.normalize_stddev.unsqueeze(0) # normalize_stddev is the specified stdev in the shape of a tensor. Unsqueeze puts it in a row
        graph_output = graph_output * normalizer
        #print('graph_output after stddev')
        #print(graph_output.shape)
        #print(graph_output)
        mean_shift = self.normalize_mean.unsqueeze(0)
        if self.normalize_atomwise:
            mean_shift = mean_shift * input_dict["num_nodes"].unsqueeze(1)
        graph_output = graph_output + mean_shift
        
        # print('graph_output after mean_shift')
        # print(graph_output.shape)
        # print(graph_output)
        
        # Apply evidential layer
        graph_output = self.evidential(graph_output)
        #print('graph_output after evidential')
        #print(graph_output.shape)
        print(graph_output)

        return graph_output
