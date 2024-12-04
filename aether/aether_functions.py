import numpy as np
import torch
import torch.nn as nn
from aether.aether_element import Element
from aether.aether_mesh import Mesh
from aether.aether_quadrature import Quadrature
from numpy.typing import NDArray

class Function(nn.Module):
    
    def __init__(self, mesh, element, bases, quadratures, device='cuda'):
        
        """
        Represents a finite element function with basis function evaluated at quadrature points.

        Parameters
        ----------
        y: ndarray
            An array  containing a set of finite element basis functions for each mesh cell evaluated
            at a set of quadrature points. y therefore has shape 
            num cells x num basis funcs x num quadrature points 
        mesh_quad : MeshQuadrature
            A MeshQuadrature object that has quadrature points in mesh coordinates. 
        func_builder : FunctionBuilder 
            A function builder object. 
        """
        super(Function, self).__init__()
        self.mesh = mesh 
        self.element = element 
        self.bases = bases
        self.quadratures = quadratures
        
        
        """
        Generate global DOF plot positions and tensors to contain DOF values. 
        """
        
        # A concatenated list of dof positions in order of vertices, edges, faces (if each dof type exists)
        dof_positions = []
        
        # Get vertex dof positions
        self.num_vertex_dofs = 0
        self.vertex_dofs_shape = (0,0)
        if element.dofs_per_vertex > 0:
            # It's possible to have multiple DOFs per vertex so repeat the coordinates along a new axis
            self.vertex_dof_positions = np.tile(self.mesh.coordinates[:, np.newaxis, :], (1, element.dofs_per_vertex, 1))
            self.vertex_dofs_shape = self.vertex_dof_positions[:,0].shape
            self.num_vertex_dofs = self.vertex_dof_positions[:,0].size
            dof_positions.append(self.vertex_dof_positions.reshape(-1,2))
        
        # Edge DOF positions
        self.num_edge_dofs = 0
        self.edge_dofs_shape = (0,0)
        if self.element.dofs_per_edge > 0:
            if element.ref_element_name == 'triangle':
                t = self.element.edge_dof_positions[2][:,0]
            elif element.ref_element_name == 'interval':
                t = self.element.edge_dof_positions[0].flatten()

            self.edge_dof_positions = mesh.edge_transform(t)
            self.edge_dofs_shape = self.edge_dof_positions[:,:,0].shape
            self.num_edge_dofs = self.edge_dof_positions[:,:,0].size
            dof_positions.append(self.edge_dof_positions.reshape(-1,2))
        
        # Face DOF positions
        self.num_face_dofs = 0
        self.face_dofs_shape = (0,0)
        if element.dofs_per_face > 0:
            self.face_dof_positions = mesh.cell_transform(element.face_dof_positions[0])
            self.face_dof_positions = np.stack([self.face_dof_positions[:,:,0], self.face_dof_positions[:,:,1]], axis=2)
            self.face_dofs_shape = self.face_dof_positions[:,:,0].shape
            self.num_face_dofs = self.face_dof_positions[:,:,0].size
            dof_positions.append(self.face_dof_positions.reshape(-1,2))
        
        self.dof_positions = np.concatenate(dof_positions)
        
        # Create DOF tensors
        if self.num_vertex_dofs > 0:
            self.vertex_dof_positions = torch.tensor(self.vertex_dof_positions, dtype=torch.float32, device=device)
            self.vertex_dofs = torch.zeros_like(self.vertex_dof_positions[:,0])
    
        if self.num_edge_dofs > 0:
            self.edge_dof_positions = torch.tensor(self.edge_dof_positions, dtype=torch.float32, device=device)
            self.edge_dofs = torch.zeros_like(self.edge_dof_positions[:,:,0])
        
        if self.num_face_dofs > 0:
            self.face_dof_positions = torch.tensor(self.face_dof_positions, dtype=torch.float32, device=device)
            self.face_dofs = torch.zeros_like(self.face_dof_positions[:,:,0])
            
                
    def get_quad_points(self, entity_dim=2):
        quad_points = []
        quad_weights = []
        
        for quad in self.quadratures[entity_dim]:
            quad_points.append(quad.quad_points)
            quad_weights.append(quad.quad_weights)
            
        quad_points = torch.stack(quad_points, dim=1)
        quad_weights = torch.stack(quad_weights, dim=1)
        
        return quad_points, quad_weights
        
        
        
class CellFunction(Function):
    
    def __init__(self, mesh : Mesh, element : Element, bases, quadratures, device='cuda'):
        
        super(CellFunction, self).__init__(mesh, element, bases, quadratures, device)
        
        self.cell_to_edges_orientation = torch.tensor(mesh.cell_to_edges_orientation, dtype=torch.int64, device=device)
        self.cell_to_vertices = torch.tensor(mesh.cell_to_vertices, dtype=torch.int64, device=device)
        self.cell_to_edges = torch.tensor(mesh.cell_to_edges[:,[1,2,0]], dtype=torch.int64, device=device)
        #self.edge_to_cells = torch.tensor(mesh.edge_to_cell, dtype=torch.int64, device=device)
        
        #self.interior_edges = torch.tensor(mesh.interior_edges, dtype=torch.int64, device=device)
        
        # A map from each edge to the cells on the +/- side of the edge
        print('a', mesh.edge_to_cells.shape)
        print('b', mesh.edge_to_cells[mesh.interior_edges].shape)
        self.interior_edge_to_cells = torch.tensor(mesh.edge_to_cells[mesh.interior_edges, :], dtype=torch.int64, device=device)
        # For the +/- side of the edge, what is the local cell edge number (e.g. global to local edge map)?
        self.interior_edge_to_cell_edge = torch.tensor(mesh.edge_to_cell_edges[mesh.interior_edges, :], dtype=torch.int64, device=device)
    
    
    def eval_interior_edges(self):
        """
        Evaluate the finite element function on +/- sides of edge and create an edge function. 
        """    
        
        F = self.forward(1)
        
        
        print(self.interior_edge_to_cells)
        print(self.interior_edge_to_cell_edge)
        
        
    
    def forward(self, entity_dim = 2):
        """
        Evaluate the finite element function at quadrature points given the degrees of freedom. 

        Returns
        -------
        tensor
            A tensor of values with the finite element function evaluated at all quadrature points.
            This tensor has shape: num cells x num quadrature points per cell.
        """
        
        
        
        # Check to make sure we have the correct bases and quadratures for this dimension
        if not entity_dim in self.bases:
            raise ValueError(
                f"To evaluate the funciton on subentities of dimension {entity_dim}"
                f"you must create the function with quadratures of that dimension."
            )
        
        local_dofs = []
        
        if self.num_vertex_dofs > 0:
            local_vertex_dofs = self.vertex_dofs[self.cell_to_vertices]
            local_vertex_dofs = local_vertex_dofs.reshape(local_vertex_dofs.shape[0], -1)
            local_dofs.append(local_vertex_dofs)
        
        if self.num_edge_dofs > 0:    
            local_edge_dofs = self.edge_dofs[self.cell_to_edges] 
            orientation = self.cell_to_edges_orientation
            local_edge_dofs = local_edge_dofs*orientation[:,:,None] + local_edge_dofs.flip(dims=(2,))*(1 - orientation[:,:,None])
            local_edge_dofs = local_edge_dofs.reshape(local_edge_dofs.shape[0], -1)
            local_dofs.append(local_edge_dofs)
        
        if self.num_face_dofs > 0:
            local_dofs.append(self.face_dofs)
            
        local_dofs = torch.column_stack(local_dofs)      
        F = []
        
        for y_i in self.bases[entity_dim]:
            f_i = local_dofs[:,:,None,None] * y_i
            f_i = f_i.sum(axis=1)
            F.append(f_i)
        
        F = torch.stack(F, dim=1)
        
        return F



class EdgeFunction(Function):
    
    def __init__(self, mesh : Mesh, element : Element, bases, quadratures, device='cuda'):
        
        super(EdgeFunction, self).__init__(mesh, element, bases, quadratures, device)
        self.edge_to_vertices = torch.tensor(mesh.edge_to_vertices, dtype=torch.int64, device=device)
        
    
    def forward(self, entity_dim = 1):
        """
        Evaluate the finite element function at quadrature points given the degrees of freedom. 

        Returns
        -------
        tensor
            A tensor of values with the finite element function evaluated at all quadrature points.
            This tensor has shape: num cells x num quadrature points per cell
        """
        
        local_dofs = []
        
        if self.num_vertex_dofs > 0:
            local_vertex_dofs = self.vertex_dofs[self.edge_to_vertices]
            local_vertex_dofs = local_vertex_dofs.reshape(local_vertex_dofs.shape[0], -1)
            local_dofs.append(local_vertex_dofs)
        
        if self.num_edge_dofs > 0:    
            local_edge_dofs = self.edge_dofs
            local_dofs.append(local_edge_dofs)
        
            
        local_dofs = torch.column_stack(local_dofs)   
        F = []
        
        for y_i in self.bases[entity_dim]:
            f_i = local_dofs[:,:,None,None] * y_i
            f_i = f_i.sum(axis=1)
            F.append(f_i)
        
        F = torch.stack(F, dim=1)
        
        return F        
