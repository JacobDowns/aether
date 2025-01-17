from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from aether.aether_element import Element
from aether.aether_mesh import Mesh
from aether.aether_quadrature import Quadrature
from jaxtyping import Float
from typing import Tuple
from typing import TYPE_CHECKING
from numpy.typing import NDArray
if TYPE_CHECKING:
    from aether.aether_function_builder import FunctionBuilder 
from typing import Optional
import pymetis
import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict
from scipy.sparse import csr_matrix, coo_matrix
from aether.aether_basis import QuadratureBasis


class Function(nn.Module):
    
    def __init__(self, func_builder : FunctionBuilder, element : Element):
        
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
        self.func_builder = func_builder
        self.mesh = func_builder.mesh  
        device = self.device = func_builder.device
        self.element = element 
        
        # Dictionary of quadrature bases
        self.quad_bases = {}
        
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
            self.vertex_dofs_shape = self.vertex_dof_positions[:,:,0].shape
            self.num_vertex_dofs = self.vertex_dof_positions[:,:,0].size
            dof_positions.append(self.vertex_dof_positions.reshape(-1,2))
        
        # Edge DOF positions
        self.num_edge_dofs = 0
        self.edge_dofs_shape = (0,0)
        if self.element.dofs_per_edge > 0:
            if element.ref_element_name == 'triangle':
                t = self.element.edge_dof_positions[2][:,0]
            elif element.ref_element_name == 'interval':
                t = self.element.edge_dof_positions[0].flatten()

            self.edge_dof_positions = self.mesh.edge_transform(t)
            self.edge_dofs_shape = self.edge_dof_positions[:,:,0].shape
            self.num_edge_dofs = self.edge_dof_positions[:,:,0].size
            dof_positions.append(self.edge_dof_positions.reshape(-1,2))
        
        # Face DOF positions
        self.num_face_dofs = 0
        self.face_dofs_shape = (0,0)
        if element.dofs_per_face > 0:
            self.face_dof_positions = self.mesh.cell_transform(element.face_dof_positions[0])
            self.face_dof_positions = np.stack([self.face_dof_positions[:,:,0], self.face_dof_positions[:,:,1]], axis=2)
            self.face_dofs_shape = self.face_dof_positions[:,:,0].shape
            self.num_face_dofs = self.face_dof_positions[:,:,0].size
            dof_positions.append(self.face_dof_positions.reshape(-1,2))
        
        self.dof_positions = np.concatenate(dof_positions)
        
        # Create DOF tensors
        if self.num_vertex_dofs > 0:
            self.vertex_dof_positions = torch.tensor(self.vertex_dof_positions, dtype=torch.float32, device=device)
            self.vertex_dofs = torch.zeros_like(self.vertex_dof_positions[:,:,0])
    
        if self.num_edge_dofs > 0:
            self.edge_dof_positions = torch.tensor(self.edge_dof_positions, dtype=torch.float32, device=device)
            self.edge_dofs = torch.zeros_like(self.edge_dof_positions[:,:,0])
        
        if self.num_face_dofs > 0:
            self.face_dof_positions = torch.tensor(self.face_dof_positions, dtype=torch.float32, device=device)
            self.face_dofs = torch.zeros_like(self.face_dof_positions[:,:,0])
    
    
    def add_basis(self, name, entity_dim : Optional[int] = None, derivatives=[]):
        element = self.element  
        func_builder = self.func_builder 
        device = func_builder.device 
        
        if entity_dim is None:
            entity_dim = element.ref_element.dimension 
        
        quad_points = []
        basis_vals = []
        for entity_index in element.ref_element.entities[entity_dim]:
            Y, mesh_quad = func_builder.eval_basis(element, entity_dim, entity_index, derivatives)
            quad_points.append(mesh_quad.quad_points)
            basis_vals.append(Y)
        
        # Shape: entity_index, subentity_index, points, 2
        quad_points = np.stack(quad_points, axis=1)
        # Shape: entity_index, subentity_index, num_basis_functions, points, range_dim
        basis_vals = np.stack(basis_vals, axis=1)
        basis_vals = torch.tensor(basis_vals, dtype=torch.float32, device=device)
        quad_weights = torch.tensor(mesh_quad.quad_weights, dtype=torch.float32, device=device)
       
        quad_basis = QuadratureBasis(quad_points, quad_weights, basis_vals, derivatives)
        self.quad_bases[name] = quad_basis
        
        
    def assemble_matrix(self, name0, name1):
        
        device = self.func_builder.device 
        
        basis0 = self.quad_bases[name0]
        basis1 = self.quad_bases[name1]
        B0 = basis0.basis 
        B1 = basis1.basis 
        
        if device == 'gpu':
            B0 = B0.cpu().numpy()
            B1 = B1.cpu().numpy()
        else:
            B0 = B0.numpy()
            B1 = B1.numpy()        
        
        element = self.element 
        N = element.num_basis_functions
        i, j = np.triu_indices(N)

        B0 = B0[:,:,i,:,:]
        B1 = B1[:,:,j,:,:]
        global_dofs0 = self.local_to_global_dofs[:,i].flatten()
        global_dofs1 = self.local_to_global_dofs[:,j].flatten()
        w = basis0.quad_weights
        
        # Integrals of basis funcs b_i * b_j 
        
        
        
        I = ((B0*B1)*w[None, None, None, :, None]).sum(axis=-2)
       
        mesh = self.mesh
        print(I.shape)
        det_A = np.absolute(mesh.cell_to_det_A)    
    
        I = np.squeeze(I*det_A[:, None, None, None])
        print(det_A)
        quit()
        I = I.flatten()

        # Make key type with two 32-bit integer items.
        key_type = types.UniTuple(types.int64, 2)
        value_type = np.float64
        N = global_dofs0.max()
        
        @njit
        def assemble(dofs0, dofs1, I):
            # Make dictionary
            d = Dict.empty(
                key_type=key_type, 
                value_type=value_type
            )
            
            for i in range(len(I)):
                d0 = dofs0[i]
                d1 = dofs1[i]
                v = I[i]
                
                key = (d0, d1)
                if key in d:
                    d[key] += v
                else:
                    d[key] = v
                
                if not d0 == d1:
                    key = (d1, d0)
                    if key in d:
                        d[key] += v
                    else:
                        d[key] = v
                        
            xi = np.zeros(len(d), dtype=np.int64)
            xj = np.zeros(len(d), dtype=np.int64)
            v = np.zeros(len(d))
            
            l = 0
            for key, value in d.items():
                i = key[0]
                j = key[1]
                xi[l] = i
                xj[l] = j 
                v[l] = value 
                l += 1  
            
            return xi, xj, v   
            
        
        xi, xj, v = assemble(global_dofs0, global_dofs1, I)
        M = coo_matrix((v, (xj, xi)), shape=(N+1,N+1))
        
        # Reorder to reduce bandwidth 
        M = M.tocsr()
        p = reverse_cuthill_mckee(M)
        M = M[p, :][:, p]
                
        return M, p
       
       
class CellFunction(Function):
    
    def __init__(self, func_builder : FunctionBuilder, element : Element):
        
        super(CellFunction, self).__init__(func_builder, element)
        
        self.cell_to_vertices = func_builder.cell_to_vertices
        self.cell_to_edges_orientation = func_builder.cell_to_edges_orientation        
        self.cell_to_edges = func_builder.cell_to_edges
        self.interior_edge_to_cells = func_builder.interior_edge_to_cells 
        self.interior_edge_to_cell_edges = func_builder.interior_edge_to_cell_edges 
        self.exterior_edge_to_cell = func_builder.exterior_edge_to_cell 
        self.exterior_edge_to_cell_edge = func_builder.exterior_edge_to_cell_edge
    
    
        # Create map from local DOFs to global DOFs
        mesh = func_builder.mesh 
        device = func_builder.device 
        
        #global_vertex_dofs = torch.tensor(self.vertex_dofs_shape, dtype=torch.int64, device=func_builder.device)
        
        local_to_global_dofs = []
        
        if element.dofs_per_vertex > 0:
            global_vertex_dofs = np.arange(self.vertex_dofs_shape[0] * self.vertex_dofs_shape[1]).reshape(self.vertex_dofs_shape)
            local_vertex_dofs = global_vertex_dofs[self.cell_to_vertices]
            local_vertex_dofs = local_vertex_dofs.reshape(local_vertex_dofs.shape[0], -1)
            local_to_global_dofs.append(local_vertex_dofs)
        
        if element.dofs_per_edge > 0:
            global_edge_dofs = np.arange(self.edge_dofs_shape[0] * self.edge_dofs_shape[1]).reshape(self.edge_dofs_shape)
            local_edge_dofs = global_edge_dofs[self.cell_to_edges] 
            orientation = self.cell_to_edges_orientation.numpy()
            local_edge_dofs = local_edge_dofs*orientation[:,:,None] + local_edge_dofs[:,:,::-1]*(1 - orientation[:,:,None])
            local_edge_dofs = local_edge_dofs.reshape(local_edge_dofs.shape[0], -1)
            local_edge_dofs += mesh.num_vertices * element.dofs_per_vertex
            local_to_global_dofs.append(local_edge_dofs)
            
        if element.dofs_per_face > 0:
            global_face_dofs = np.arange(self.face_dofs_shape[0] * self.face_dofs_shape[1]).reshape(self.face_dofs_shape)
            local_face_dofs = global_face_dofs
            local_face_dofs += mesh.num_vertices * element.dofs_per_vertex + mesh.num_edges * element.dofs_per_edge
            local_to_global_dofs.append(local_face_dofs)
        
        self.local_to_global_dofs = np.column_stack(local_to_global_dofs)
            
            #local_vertex_dofs = local_vertex_dofs.reshape(local_vertex_dofs.shape[0], -1)
        
        #print(global_vertex_dofs)
        
        #local_vertex_dofs = self.vertex_dofs[self.cell_to_vertices]
        #local_vertex_dofs = local_vertex_dofs.reshape(local_vertex_dofs.shape[0], -1)
        #local_dofs.append(local_vertex_dofs)
        
        
        
    def eval_interior_edges(self, side='+') ->  Tuple[
        Float[torch.tensor, 'entity sub_entity quad_point range_dim'],
        Float[torch.tensor, 'quad_point'],
        Float[torch.tensor, 'entity sub_entity quad_point coordinate']
    ]:
        """
        Evaluate the finite element function the + or - side of each interior edge. 
        """    
        
        X, W, F = self.forward(1)
        side_dict = {'+' : 0, '-' : 1}
        index = side_dict[side]
        E = F[self.interior_edge_to_cells[:,index], self.interior_edge_to_cell_edges[:,index], :, :]
        X = X[self.interior_edge_to_cells[:,index], self.interior_edge_to_cell_edges[:,index], :, :]
        X = X[:,None,:,:]
        F = E[:,None,:,:]
        
        return X, W, F
    
    
    def eval_exterior_edges(self) -> Tuple[
        Float[torch.tensor, 'entity sub_entity quad_point range_dim'],
        Float[torch.tensor, 'quad_point'],
        Float[torch.tensor, 'entity sub_entity quad_point coordinate']
    ]:
        """
        Evaluate the finite element function on each exterior edge. 
        """ 
        
        X, W, F = self.forward(1)
        E = F[self.exterior_edge_to_cell, self.exterior_edge_to_cell_edge, :, :]
        X = X[self.exterior_edge_to_cell, self.exterior_edge_to_cell_edge, :, :]
        X = X[:,None,:,:]
        F = E[:,None,:,:]
        
        return X, W, F 
    
    
    def eval_cells(self):
        """
        Evaluate the finite element function on each cell. 
        """ 
        
        X, W, F = self.forward(2)
        return X, W, F
    
    
    def forward(self, entity_dim = 2) -> Tuple[
        Float[torch.tensor, 'entity sub_entity quad_point range_dim'],
        Float[torch.tensor, 'quad_point'],
        Float[torch.tensor, 'entity sub_entity quad_point coordinate']
    ]:
        
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
        
        X, W = self.get_quad_points(entity_dim)
        F = torch.stack(F, dim=1)
        
        return X, W, F



class EdgeFunction(Function):
    
    def __init__(self, func_builder : FunctionBuilder, element : Element):
        
        super(EdgeFunction, self).__init__(func_builder, element)
        
        self.edge_to_vertices = func_builder.edge_to_vertices
        self.interior_edges = func_builder.interior_edges 
        self.exterior_edges = func_builder.exterior_edges
        
        
    def eval_interior_edges(self)  -> Tuple[
        Float[torch.tensor, 'entity sub_entity quad_point range_dim'],
        Float[torch.tensor, 'quad_point'],
        Float[torch.tensor, 'entity sub_entity quad_point coordinate']
    ]:
        X, W, F = self.forward(1)
        X = X[self.interior_edges]
        F = F[self.interior_edges]
        
        return X, W, F 
    
    
    def eval_exterior_edges(self) -> Tuple[
        Float[torch.tensor, 'entity sub_entity quad_point range_dim'],
        Float[torch.tensor, 'quad_point'],
        Float[torch.tensor, 'entity sub_entity quad_point coordinate']
    ]:  
        X, W, F = self.forward(1)
        X = X[self.exterior_edges]
        F = F[self.exterior_edges]
        
        return X, W, F 
    
    
    def forward(self, entity_dim = 1) -> Tuple[
        Float[torch.tensor, 'entity sub_entity quad_point range_dim'],
        Float[torch.tensor, 'quad_point'],
        Float[torch.tensor, 'entity sub_entity quad_point coordinate']
    ]:
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
        X, W = self.get_quad_points(entity_dim)
        
        return X, W, F
