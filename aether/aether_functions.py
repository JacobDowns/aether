import numpy as np
import torch
import torch.nn as nn
import itertools
from aether.aether_element import AetherElement
from aether.aether_mesh import Mesh
from aether.aether_quadrature import ReferenceQuadrature, MeshQuadrature

class QuadratureFunction(nn.Module):
    
    def __init__(self, y, mesh_quad : MeshQuadrature, func_builder : 'FunctionBuilder', device='cuda'):
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
    
        super(QuadratureFunction, self).__init__()
        self.mesh_quad = mesh_quad
        self.func_builder = func_builder 

        self.x = torch.tensor(self.mesh_quad.mesh_quad_points, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)
        
        self.edge_orientation = torch.tensor(func_builder.mesh.faces_to_edge_orientation, dtype=torch.int64, device=device)
        self.faces = torch.tensor(func_builder.mesh.faces, dtype=torch.int64, device=device)
        self.faces_to_edges = torch.tensor(func_builder.mesh.faces_to_edges[:,[1,2,0]], dtype=torch.int64, device=device)
        
        self.num_vertex_dofs = func_builder.num_vertex_dofs
        self.num_edge_dofs = func_builder.num_edge_dofs
        self.num_face_dofs = func_builder.num_face_dofs
        
        self.vertex_dofs_shape = func_builder.vertex_dofs_shape
        self.edge_dofs_shape = func_builder.edge_dofs_shape
        self.face_dofs_shape = func_builder.face_dofs_shape
        
        if self.num_vertex_dofs > 0:
            self.vertex_dof_positions = func_builder.vertex_dof_positions
        
        if self.num_edge_dofs > 0:
            self.edge_dof_positions = func_builder.edge_dof_positions
        
        if self.num_face_dofs > 0:
            self.face_dof_positions = func_builder.face_dof_positions
        
        
    def forward(self, **kwargs):
        """
        Evaluate the finite element function at quadrature points given the degrees of freedom. 

        Keyword Agruments:
        ----------
        vertex_dofs : tensor
            A tensor of values for all vertex dofs. The shape of this tensor is given by 
            num cells x 3 x dofs per vertex
            where the second dimension is the vertex number. 
        edge_dofs : tensor
            A tensor of values for all vertex dofs. The shape of this tensor is given by 
            num cells x 3 x dofs per edge
            where dimension 2 is the edge number.
        face_dofs : tensor 
            A tensor of values for all face dofs. The shape of this tensor is given by 
            num cells x dofs per face
    
        Returns
        -------
        tensor
            A tensor of values with the finite element function evaluated at all quadrature points.
            This tensor has shape 
            num cells x num quadrature points per cell
        """
        
        local_dofs = []
        
        if self.num_vertex_dofs > 0:
            vertex_dofs = kwargs['vertex_dofs']
            local_vertex_dofs = vertex_dofs[self.faces]
            local_vertex_dofs = local_vertex_dofs.reshape(local_vertex_dofs.shape[0], -1)
            local_dofs.append(local_vertex_dofs)
        
        if self.num_edge_dofs > 0:    
            edge_dofs = kwargs['edge_dofs']
            local_edge_dofs = edge_dofs[self.faces_to_edges] 
            orientation = self.edge_orientation
            local_edge_dofs = local_edge_dofs*orientation[:,:,None] + local_edge_dofs.flip(dims=(2,))*(1 - orientation[:,:,None])
            local_edge_dofs = local_edge_dofs.reshape(local_edge_dofs.shape[0], -1)
            local_dofs.append(local_edge_dofs)
        
        if self.num_face_dofs > 0:
            local_face_dofs = kwargs['face_dofs']
            local_dofs.append(local_face_dofs)
        
        # Compute weighted sums of basis functions
        local_dofs = torch.column_stack(local_dofs)        
        f = local_dofs[:,:,None] * self.y
        f = f.sum(axis=1)
    
        return f


class FunctionBuilder:
      
    def __init__(self, mesh : Mesh, element : AetherElement):
            
        """
        Object that is used to create quadrature functions for a Lagrange element of given degree on a mesh.

        Parameters
        ----------
        mesh : Mesh
            A mesh object.
        degree: int
            Degree of Lagrange element
        """
        
        self.mesh = mesh
        self.aether_element = element
        self.symfem_element = element.element 
        
        # A concatenated list of dof positions in order of vertices, edges, faces (if each dof type exists)
        dof_positions = []
        
        # Get vertex dof positions
        self.num_vertex_dofs = 0
        self.vertex_dofs_shape = (0,0)
        if self.aether_element.dofs_per_vertex > 0:
            self.vertex_dof_positions = self.mesh.coordinates
            self.vertex_dofs_shape = self.vertex_dof_positions[:,0].shape
            self.num_vertex_dofs = self.vertex_dof_positions[:,0].size
            dof_positions.append(self.vertex_dof_positions)
        
        # Edge dof positions
        self.num_edge_dofs = 0
        self.edge_dofs_shape = (0,0)
        if self.aether_element.dofs_per_edge > 0:
            t = self.aether_element.edge_dof_positions[:,0]
            coords0 = mesh.coordinates[mesh.edges_to_vertices[:,0]]
            coords1 = mesh.coordinates[mesh.edges_to_vertices[:,1]]
            self.edge_dof_positions = coords1[:,np.newaxis,:]*t[np.newaxis,:,np.newaxis] + coords0[:,np.newaxis,:]*(1.-t[np.newaxis,:,np.newaxis])
            self.edge_dofs_shape = self.edge_dof_positions[:,:,0].shape
            self.num_edge_dofs = self.edge_dof_positions[:,:,0].size
            dof_positions.append(self.edge_dof_positions.reshape(-1,2))
        
        # Face dof positions
        self.num_face_dofs = 0
        self.face_dofs_shape = (0,0)
        if self.aether_element.dofs_per_face > 0:
            self.face_dof_positions = np.matmul(self.mesh.A, self.aether_element.face_dof_positions.T) + self.mesh.B[:,:,np.newaxis]
            self.face_dof_positions = np.stack([self.face_dof_positions[:,0,:], self.face_dof_positions[:,1,:]], axis=2)
            self.face_dofs_shape = self.face_dof_positions[:,:,0].shape
            self.num_face_dofs = self.face_dof_positions[:,:,0].size
            dof_positions.append(self.face_dof_positions.reshape(-1,2))
        
        self.dof_positions = np.concatenate(dof_positions)
                
            
    def get_function(self, quadrature : ReferenceQuadrature, *args):
        
        """
        Returns a quadrature function for a set of quadrature points. Derivatives for each coordinate dimension
        can be passed in as additional arguments. 

        Parameters
        ----------
        quad_points : ndarray
            A set of quadrature points defined on the reference element of shape  
            num quadrature points x 2
        
        
        Additional Arguments
        ----------
        (d1, d2 ...)
            An optional, variable length tuple of 0 or 1 values. 0 denotes a derivative in the x direction, 1 in the y direction.
            For instance, to take the x y derivative input 0 1 or 1 1 for the y y derivative. 
            
        Returns
        -------
        QuadratureFunction
            A quadrature function used to evaluate a finite element function at quadrature points given DOF values. 
            
        """
        
        quad_points = quadrature.quad_points

        # Evaluate all basis functions on each mesh cell. 
        Y = []
        for c in itertools.product(*([[0, 1]]*len(args))):
            w = np.prod(self.mesh.A_inv[:,c,args], axis=1)
            du = self.aether_element.eval_func(quad_points, *c)            
            yi = w[:,np.newaxis,np.newaxis] * du
            Y.append(yi)
        
        y = np.array(Y).sum(axis=0)
        
        # Extend quadrature to whole mesh 
        mesh_quad = MeshQuadrature(self.mesh, quadrature) 
    
        f = QuadratureFunction(y, mesh_quad, self)
        return f