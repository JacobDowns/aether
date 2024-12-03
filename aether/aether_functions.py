import numpy as np
import torch
import torch.nn as nn
import itertools
from aether.aether_element import Element
from aether.aether_mesh import Mesh, TorchMesh
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
            self.vertex_dof_positions = self.mesh.coordinates
            self.vertex_dofs_shape = self.vertex_dof_positions[:,0].shape
            self.num_vertex_dofs = self.vertex_dof_positions[:,0].size
            dof_positions.append(self.vertex_dof_positions)
        
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
            self.face_dof_positions = np.stack([self.face_dof_positions[:,0,:], self.face_dof_positions[:,1,:]], axis=2)
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
            
                
            
        #self.x = torch.tensor(self.mesh_quad.mesh_quad_points, dtype=torch.float32, device=device)
        #self.y = torch.tensor(y, dtype=torch.float32, device=device)
        
        #self.edge_orientation = torch.tensor(func_builder.mesh.faces_to_edge_orientation, dtype=torch.int64, device=device)
        #self.faces = torch.tensor(func_builder.mesh.faces, dtype=torch.int64, device=device)
        #self.faces_to_edges = torch.tensor(func_builder.mesh.faces_to_edges[:,[1,2,0]], dtype=torch.int64, device=device)
        
    def forward(self):
       pass
        
class CellFunction(Function):
    
    def __init__(self, mesh : TorchMesh, element : Element, bases, quadratures, device='cuda'):
        
        super().__init__(element, bases, quadratures, device)
        
        self.mesh = mesh 
        self.element = element 
        self.cell_to_edges_orientation = mesh.cell_to_edges_orientation.to(device)
        self.cell_to_vertices = mesh.cell_to_vertices.to(device)
        self.cell_to_edges = mesh.cell_to_edges[:,[1,2,0]].to(device)
        
    
    def forward(self, entity_dim = 2):
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
            
        # Compute weighted sums of basis functions
        local_dofs = torch.column_stack(local_dofs)        
        
        if entity_dim == 2:
            y = self.bases[entity_dim]
            f = local_dofs[:,:,None] * self.y
            f = f.sum(axis=1)
        else:
            """
            If we're evaluating the function on edges or vertices, then append the function 
            evaluated on each subeneity.
            """
            for y_i in self.bases[entity_dim]:
                f_i = local_dofs[:,:,None] * self.y
                f_i = f.sum(axis=1)
        
    
        return f