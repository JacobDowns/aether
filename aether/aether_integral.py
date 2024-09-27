import torch
import torch.nn as nn
from torch_scatter import scatter
from aether_functions import QuadratureFunction

class TestIntegral(nn.Module):
    
    def __init__(self, v : QuadratureFunction):
        """
        This module performs integrals of a quadrature function with a set of test functions.

        Parameters
        ----------
        v : QuadratureFunction
            A quadrature function whose basis forms the test functions.
        quad_weights: tensor
            A
        """
        
        super(TestIntegral, self).__init__()
        
        self.v = v

        mesh = v.func_builder.mesh
        func_builder = v.func_builder
        
        # Necessary mesh information for computing integrals
        self.det_A = torch.tensor(mesh.det_A, dtype=torch.float32)
        self.faces = torch.tensor(mesh.faces, dtype=torch.int64)
        #self.faces_to_edges = torch.tensor(mesh.faces_to_edges, dtype=torch.int64)
        self.faces_to_edges = torch.tensor(func_builder.mesh.faces_to_edges[:,[1,2,0]], dtype=torch.int64)
        self.faces_to_edge_orientation = torch.tensor(mesh.faces_to_edge_orientation, dtype=torch.int64)
        
        # These are basis function evaluated at quadrature points
        self.v_x = v.y
        # Quadrature weights
        self.quad_weights = v.quad_weights
        
        self.dofs_per_vertex = func_builder.torch_lagrange.dofs_per_vertex
        self.dofs_per_edge = func_builder.torch_lagrange.dofs_per_edge
        self.dofs_per_face = func_builder.torch_lagrange.dofs_per_face
        
        
    def forward(self, f_x):
        """
        Takes in a function f evaluated at quadrature points f_x and computes integrals of f*v_i
        for all test functions v_i. 

        Keyword Agruments:
        ----------
        f_x : tensor
            A tensor of the function f evaluated at quadrature points. Has shape
            num cells x num quadrature points
    
        Returns
        -------
        Returns:
        dict: A dictionary that possibly contains the following keys:
            - 'vertex_dofs' (tensor): Returns integrals of f*v_i for vertex dofs
            - 'fedge_dofs' (tensor): Returns integrals of f*v_i for edge dofs
            - 'face_dofs' (tensor): Returns integrals of f*v_i for face dofs
        """

        ### Compute per cell integrals
        ######################################################################
        
        v_x = self.v_x
        # For each cell multiply the function f at quadrature points x by each of the basis function in v
        # evaluated at quadrature points x
        I_x = f_x[:, None, :]*v_x
        # Compute weighted sum of function at quadrature points
        I_x = I_x * self.quad_weights[None, None, :]
        I_x = I_x.sum(axis=2) * self.det_A[:, None]
        
        d = {}
        
        ### Compute integrals for "vertex test functions" (DOFs on vertex)
        ######################################################################
        
        # Vertex test functions span numerous cells, so we need to accumulate all the appropriate cell integrals
        if self.dofs_per_vertex > 0:
            # Get all cell integrals associated with vertices
            I_x_vertex = I_x[:, 0:(3*self.dofs_per_vertex)]
            vertex_dofs = scatter(I_x_vertex.flatten(), self.faces.flatten(), dim=0, reduce='add')
            d['vertex_dofs'] = vertex_dofs
            
        
        ### Compute integrals for "edge test functions" (DOFs on edge)
        ######################################################################

        if self.dofs_per_edge > 0:
         
            edge_start = 3*self.dofs_per_vertex
            edge_end = edge_start + 3*self.dofs_per_edge
            I_x_edge = I_x[:,edge_start:edge_end]
            I_x_edge = I_x_edge.reshape(I_x_edge.shape[0], 3, -1)

            # Make sure edge DOFs are appropriately oriented
            I_x_edge = I_x_edge * self.faces_to_edge_orientation[:,:,None] + (1 - self.faces_to_edge_orientation[:,:,None])*I_x_edge.flip(dims=(2,))
               
            indexes = self.faces_to_edges[:,:]
            I_x_edge = I_x_edge.reshape(I_x_edge.shape[0]*3,-1)
            indexes = indexes.flatten()
            edge_dofs = scatter(I_x_edge, indexes, dim=0, reduce='add')
            d['edge_dofs'] = edge_dofs
        
        
        ### Integrals for "face test functions" (DOFs on face)
        ######################################################################
        
        if self.dofs_per_face > 0: 
            # We did it!
            face_dofs = I_x[:,edge_end:]
            d['face_dofs'] = face_dofs
        
        
        return d
    
    
        