import torch
import torch.nn as nn
from torch_scatter import scatter
from aether.aether_functions import QuadratureFunction, FunctionBuilder

class Decoder(torch.nn.Module):
    def __init__(self,n_features=64,n_hidden=64,n_out=2,activation=torch.nn.Tanh()):    
        super(Decoder,self).__init__()
        self.decoder = build_mlp(n_features,n_hidden,n_out,lay_norm=False)

    def forward(self,x):
        x = self.decoder(x)
        return x

class EdgeBlock(torch.nn.Module):
    def __init__(self, custom_func=None):
        
        super(EdgeBlock, self).__init__()
        self.net = custom_func

    def forward(self, x_node, x_edge):

        # Concatenate values from node endpoints to current edge values.
        x_edge_ = torch.cat([
            x_node[m.dual_edges.T[:,0]],
            x_node[m.dual_edges.T[:,1]],
            x_edge
        ], dim=1)

        # Apply an MLP (or custom function)
        x_edge_ = self.net(x_edge_)  

        return x_edge_



class NodeBlock(torch.nn.Module):

    def __init__(self, custom_func=None):

        super(NodeBlock, self).__init__()

        self.net = custom_func

    def forward(self, x_node, x_edge):

        # Aggregate edge features
        x_node1 = torch.zeros((x_node.shape[0], x_edge.shape[1]), device=x_node.device)
        
        torch_scatter.scatter_add(x_edge, m.dual_edges.T[:,0], dim=0, out=x_node1)
        torch_scatter.scatter_add(x_edge, m.dual_edges.T[:,1], dim=0, out=x_node1)

        # Concatenate with current node values
        x_node_ = torch.cat([
            x_node,
            x_node1
        ], dim=1)
        
        # Apply an MLP (or other defined function)
        x_node_ = self.net(x_node_)
        return x_node_

def build_mlp(in_size, hidden_size, out_size, lay_norm=True):

    module = torch.nn.Sequential(
        torch.nn.Linear(in_size, hidden_size),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.Tanh(),
        #torch.nn.Linear(hidden_size, hidden_size),
        #torch.nn.Tanh(),
        torch.nn.Linear(hidden_size, out_size)
    )
    if lay_norm: return torch.nn.Sequential(module,  torch.nn.LayerNorm(normalized_shape=out_size))
    return module

class Encoder(torch.nn.Module):
    def __init__(self,n_features=1,n_hidden=64,n_out=64,activation=torch.nn.Tanh()):

        super(Encoder,self).__init__()
        self.encoder = build_mlp(n_features,n_hidden,n_out,lay_norm=True)

    def forward(self,x):
        x = self.encoder(x)
        return x

class EncoderProcessorDecoder(torch.nn.Module):
    def __init__(self, message_passing_num, edges):

        super(EncoderProcessorDecoder, self).__init__()

        # Define encoders
        self.cell_encoder = Encoder(n_features=2,n_hidden=64,n_out=64)
        self.edge_encoder = Encoder(n_features=4,n_hidden=64,n_out=64)

        # Define message passing methods
        self.processer_list_nodes = []
        self.processer_list_edges = []
        for _ in range(message_passing_num):
            self.processer_list_nodes.append(NodeBlock(custom_func=build_mlp(2*64,64,64)))
            self.processer_list_edges.append(EdgeBlock(custom_func=build_mlp(3*64,64,64)))
        self.processer_list_nodes = torch.nn.ModuleList(self.processer_list_nodes) 
        self.processer_list_edges = torch.nn.ModuleList(self.processer_list_edges) 

        # Define decoder
        self.decoder = Decoder()

    def forward(self, x_cell, x_edge):

        # Encode cell and edge features
        h_cell = self.cell_encoder(x_cell)
        h_edge = self.edge_encoder(x_edge)

        # Perform message passing
        for nb,eb in zip(self.processer_list_nodes,self.processer_list_edges):
            h_cell = h_cell + nb(h_cell,h_edge)
            h_edge = h_edge + eb(h_cell,h_edge)

        # Decode
        decoded = self.decoder(h_cell)

        return decoded