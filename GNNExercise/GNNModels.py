"""Simple invariant and equivariant graph neural networks."""
import torch


class GNNInvariant(torch.nn.Module):
    """Translation and rotation invariant graph neural network.

    Keyword Arguments
    -----------------
        output_dim : Dimension of output (default 2)
        state_dim : Dimension of the node states (default 10)
        num_message_passing_rounds : Number of message passing rounds
            (default 3)
    """

    def __init__(self, output_dim=2, state_dim=10,
                 num_message_passing_rounds=3):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.state_dim = state_dim
        self.edge_dim = 1
        self.output_dim = output_dim
        self.num_message_passing_rounds = num_message_passing_rounds

        # Message passing networks
        self.message_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim+self.edge_dim, self.state_dim),
            torch.nn.Tanh())

        # State output network
        self.output_net = torch.nn.Linear(
            self.state_dim, self.output_dim)

    def forward(self, x):
        """Evaluate neural network on a batch of graphs.

        Parameters
        ----------
        x: GraphDataset
            A data set object that contains the following member variables:

            node_coordinates : torch.tensor (N x 2)
                2d coordinates for each of the N nodes in all graphs
            node_graph_index : torch.tensor (N)
                Index of which graph each node belongs to
            edge_list : torch.tensor (E x 2)
                Edges (to-node, from-node) in all graphs
            node_to, node_from : Shorthand for column 0 and 1 in edge_list
            edge_lengths : torch.tensor (E)
                Edge features

        Returns
        -------
        out : N x output_dim
            Neural network output

        """
        # Initialize node features to zeros
        self.state = torch.zeros([x.num_nodes, self.state_dim])

        # Loop over message passing rounds
        for _ in range(self.num_message_passing_rounds):
            # Input to message passing networks
            inp = torch.cat((self.state[x.node_from], x.edge_lengths), 1)

            # Message networks
            message = self.message_net(inp)

            # Aggregate: Sum messages
            self.state.index_add_(0, x.node_to, message)

        # Aggretate: Sum node features
        self.graph_state = torch.zeros((x.num_graphs, self.state_dim))
        self.graph_state.index_add_(0, x.node_graph_index, self.state)

        # Output
        out = self.output_net(self.graph_state)
        return out


class GNNEquivariant(torch.nn.Module):
    """Translation and rotation invariant graph neural network.

    Keyword Arguments
    -----------------
        output_dim : Dimension of output (default 7)
        state_dim : Dimension of the node states (default 10)
        num_message_passing_rounds : Number of message passing rounds
            (default 3)
    """

    def __init__(self, output_dim=2, state_dim=10,
                 num_message_passing_rounds=3):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.state_dim = state_dim
        self.edge_dim = 1
        self.output_dim = output_dim
        self.num_message_passing_rounds = num_message_passing_rounds

        # Message passing networks
        self.message_net_dot = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim+self.edge_dim, self.state_dim),
            torch.nn.Tanh())
        self.message_net_cross = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim+self.edge_dim, self.state_dim),
            torch.nn.Tanh())
        self.message_net_vector = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim+self.edge_dim, self.state_dim),
            torch.nn.Tanh())

        # State output network
        self.output_net = torch.nn.Linear(
            self.state_dim, self.output_dim)

    def forward(self, x):
        """Evaluate neural network on a batch of graphs.

        Parameters
        ----------
        x: GraphDataset
            A data set object that contains the following member variables:

            node_coordinates : torch.tensor (N x 2)
                2d coordinates for each of the N nodes in all graphs
            node_graph_index : torch.tensor (N)
                Index of which graph each node belongs to
            edge_list : torch.tensor (E x 2)
                Edges (to-node, from-node) in all graphs
            node_to, node_from : Shorthand for column 0 and 1 in edge_list
            edge_lengths : torch.tensor (E)
                Edge features
            edge_vectors : torch.tensor (E x 2)
                Edge vector features

        Returns
        -------
        out : N x output_dim
            Neural network output

        """
        # Initialize node features to zeros
        self.state = torch.zeros([x.num_nodes, self.state_dim])
        self.state_vec = torch.zeros([x.num_nodes, 2, self.state_dim])

        # Loop over message passing rounds
        for _ in range(self.num_message_passing_rounds):
            # Input to message passing networks
            inp = torch.cat((self.state[x.node_from], x.edge_lengths), 1)

            # Compute dot and cross product
            dot_product = dot(
                self.state_vec[x.node_from], self.state_vec[x.node_to])
            cross_product = cross(
                self.state_vec[x.node_from], self.state_vec[x.node_to])

            # Message networks
            message = (self.message_net_dot(inp) * dot_product +
                       self.message_net_cross(inp) * cross_product)
            message_vec = (self.message_net_vector(inp)[:, None, :] *
                           x.edge_vectors[:, :, None])

            # Aggregate: Sum messages
            self.state.index_add_(0, x.node_to, message)
            self.state_vec.index_add_(0, x.node_to, message_vec)

        # Aggretate: Sum node features
        self.graph_state = torch.zeros((x.num_graphs, self.state_dim))
        self.graph_state.index_add_(0, x.node_graph_index, self.state)

        # Output
        out = self.output_net(self.graph_state)
        return out


def cross(v1, v2):
    """Compute the 2-d cross product.

    Parameters
    ----------
    v1, v2 : Array
        Arrays (shape Nx2) containing N 2-d vectors

    Returns
    -------
    Array
        Array (shape N) containing the cross products

    """
    return v1[:, 0]*v2[:, 1] - v1[:, 1]*v2[:, 0]


def dot(v1, v2):
    """Compute the 2-d dot product.

    Parameters
    ----------
    v1, v2 : Array
        Arrays (Nx2) containing N 2-d vectors

    Returns
    -------
    Array
        Array (shape N) containing the dot products

    """
    return v1[:, 0]*v2[:, 0] + v1[:, 1]*v2[:, 1]
