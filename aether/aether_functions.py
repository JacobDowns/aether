import numpy as np
import torch
import torch.nn as nn
import itertools
from aether.aether_element import Element
from aether.aether_mesh import Mesh
from aether.aether_quadrature import ReferenceQuadrature, MeshQuadrature, TriangleEdgeQuadrature, TriangleQuadrature, IntervalQuadrature
from numpy.typing import NDArray

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
        
        """
        DOF Tensors:
        
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
        """
        
        if self.num_vertex_dofs > 0:
            self.vertex_dof_positions = torch.tensor(func_builder.vertex_dof_positions, dtype=torch.float32, device=device)
            self.vertex_dofs = torch.zeros_like(self.vertex_dof_positions[:,0])
        
        if self.num_edge_dofs > 0:
            self.edge_dof_positions = torch.tensor(func_builder.edge_dof_positions, dtype=torch.float32, device=device)
            self.edge_dofs = torch.zeros_like(self.edge_dof_positions[:,:,0])
        
        if self.num_face_dofs > 0:
            self.face_dof_positions = torch.tensor(func_builder.face_dof_positions, dtype=torch.float32, device=device)
            self.face_dofs = torch.zeros_like(self.face_dof_positions[:,:,0])
        
        
    def forward(self):
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
            local_vertex_dofs = self.vertex_dofs[self.faces]
            local_vertex_dofs = local_vertex_dofs.reshape(local_vertex_dofs.shape[0], -1)
            local_dofs.append(local_vertex_dofs)
        
        if self.num_edge_dofs > 0:    
            local_edge_dofs = self.edge_dofs[self.faces_to_edges] 
            orientation = self.edge_orientation
            local_edge_dofs = local_edge_dofs*orientation[:,:,None] + local_edge_dofs.flip(dims=(2,))*(1 - orientation[:,:,None])
            local_edge_dofs = local_edge_dofs.reshape(local_edge_dofs.shape[0], -1)
            local_dofs.append(local_edge_dofs)
        
        if self.num_face_dofs > 0:
            local_dofs.append(self.face_dofs)
        
        # Compute weighted sums of basis functions
        local_dofs = torch.column_stack(local_dofs)        
        f = local_dofs[:,:,None] * self.y
        f = f.sum(axis=1)
    
        return f


class FunctionBuilder:
      
    def __init__(self, mesh : Mesh, element : Element):
            
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
            if self.aether_element.ref_element == 'triangle':
                t = self.aether_element.edge_dof_positions[2][:,0]
            elif self.aether_element.ref_element == 'interval':
                t = self.aether_element.edge_dof_positions[0].flatten()

            self.edge_dof_positions = mesh.edge_transform(t)
            self.edge_dofs_shape = self.edge_dof_positions[:,:,0].shape
            self.num_edge_dofs = self.edge_dof_positions[:,:,0].size
            dof_positions.append(self.edge_dof_positions.reshape(-1,2))
        
        # Face dof positions
        self.num_face_dofs = 0
        self.face_dofs_shape = (0,0)
        if self.aether_element.dofs_per_face > 0:
            self.face_dof_positions = mesh.cell_transform(self.aether_element.face_dof_positions[0])
            self.face_dof_positions = np.stack([self.face_dof_positions[:,0,:], self.face_dof_positions[:,1,:]], axis=2)
            self.face_dofs_shape = self.face_dof_positions[:,:,0].shape
            self.num_face_dofs = self.face_dof_positions[:,:,0].size
            dof_positions.append(self.face_dof_positions.reshape(-1,2))
        
        self.dof_positions = np.concatenate(dof_positions)
            
    
    def contravariant_piola_transform(self, y : NDArray):
        """
        Given an N x J x K x 2 array, perform a contravariant Piola transform.  
        """
        
        mesh = self.mesh 
        W = (mesh.cell_to_det_A)[:,np.newaxis,np.newaxis] * mesh.cell_to_A
        y = np.einsum('nij,nlkj->nlki', W, y)
        return y 
    
    
    def covariant_piola_transform(self, y : NDArray):
        """
        Given an N x J x K x 2 array, perform a covariant Piola transform.  
        """
        
        mesh = self.mesh 
        W = np.transpose(mesh.cell_to_A_inv, axes=(0,2,1))
        y = np.einsum('nij,nlkj->nlki', W, y)
        return y 
    
            
    def get_function(self, quadrature : ReferenceQuadrature, derivatives = [], transform='affine'):
        
        """
        Returns a quadrature function for a set of quadrature points. Derivatives for each coordinate dimension
        can be passed in as additional arguments. 

        Parameters
        ----------
        quad_points : ndarray
            A set of quadrature points defined on the reference element of shape  
            num quadrature points x 2
        derivatives : list of strings
            A list of derivatives to take for each basis function. For example, use eval_func(points, ['x', 'y']) to get the xy partial 
            derivatives. Can be left blank for no derivatives. 
            
        Returns
        -------
        QuadratureFunction
            A quadrature function used to evaluate a finite element function at quadrature points given DOF values. 
            
        """
        
        quad_points = quadrature.quad_points
        
        if self.aether_element.ref_element == 'triangle':
            indexes = [[0,1]]
        elif self.aether_element.ref_element == 'interval':
            indexes=[[0]]
        
        # Dimension of the range
        D = self.aether_element.range_dim
        # If derivatives aren't specified, just initialize empty lists for each range dimension
        if len(derivatives) == 0:
            derivatives = [[] for i in range(D)]

        symbol_dict = {'x' : 0, 'y' : 1}
        
        Y = []
        for d in range(D):
            
            ds = derivatives[d]
            ds_indexes = [symbol_dict[s] for s in ds]

            # Evaluate all basis functions on each mesh cell. 
            y_d = []
            
            """
            Evaluating derivatives in physical coordinates requires some somwehat unpleasant 
            chain ruling. See:
            https://scicomp.stackexchange.com/questions/25196/implementing-higher-order-derivatives-for-finite-element 
            """
            if self.aether_element.ref_element == 'triangle':
                
                A_inv = self.mesh.cell_to_A_inv
                for coord_dim in itertools.product(*(indexes*len(ds))):
                    # Weights are products of entries of the transform matrix
                    w = np.prod(A_inv[:, coord_dim, ds_indexes], axis=1)
                    du = self.aether_element.eval_basis(quad_points, ds, d=d)            
                    yi = w[:,np.newaxis,np.newaxis] * du
                    y_d.append(yi)
                
                y_d = np.array(y_d).sum(axis=0)
                
            elif self.aether_element.ref_element == 'interval':
                # The 1d transformation case
                du = self.aether_element.eval_basis(quad_points, ds, d=d)
                w = (1. / self.mesh.edge_to_length)**len(ds)
                y_d = w[:,np.newaxis,np.newaxis] * du
                #print(yi.shape)
            
            Y.append(y_d)
        
        """
        Combine any vector outputs into a single array. The resulting dimension of the array 
        is N x J x K x D. N is the number of cells for an element defined on a cell, or 
        the number of edges for an element defined on an interval. J is the number of of basis 
        functions per mesh entity (edge or cell). K is the number of quadrature points.  D is the 
        dimension of the range of each basis function (e.g. 1 for a scalar basis function or 2)
        for a vector basis function. 
        """
        Y = np.stack(Y, axis=-1)
        
        # For H(div) elements use a contravariant Piola transform
        if self.aether_element.continuity == 'H(div)':
            Y = self.contravariant_piola_transform(Y)
        elif self.aether_element.continuity == 'H(curl)':
            # For H(curl) elements use a covariant Piola transform
            Y = self.covariant_piola_transform(Y)
            
        # Extend quadrature to entire mesh 
        mesh_quad = MeshQuadrature(self.mesh, quadrature) 
    
        return Y, mesh_quad
        #f = QuadratureFunction(Y, mesh_quad, self)
        #return f
        
        
    def eval_basis(
        self, 
        quad_points : NDArray,
        derivatives = []
    ):
        """
        Evaluates all finite element basis functions at a set of quadrature points.
        """
        
        if self.aether_element.ref_element == 'triangle':
            indexes = [[0,1]]
        elif self.aether_element.ref_element == 'interval':
            indexes=[[0]]
        
        # Dimension of the range
        D = self.aether_element.range_dim
        # If derivatives aren't specified, just initialize empty lists for each range dimension
        if len(derivatives) == 0:
            derivatives = [[] for i in range(D)]

        symbol_dict = {'x' : 0, 'y' : 1}
        
        Y = []
        for d in range(D):
            
            ds = derivatives[d]
            ds_indexes = [symbol_dict[s] for s in ds]

            # Evaluate all basis functions on each mesh cell. 
            y_d = []
            
            """
            Evaluating derivatives in physical coordinates requires some somwehat unpleasant 
            chain ruling. See:
            https://scicomp.stackexchange.com/questions/25196/implementing-higher-order-derivatives-for-finite-element 
            """
            if self.aether_element.ref_element == 'triangle':
                
                A_inv = self.mesh.cell_to_A_inv
                for coord_dim in itertools.product(*(indexes*len(ds))):
                    # Weights are products of entries of the transform matrix
                    w = np.prod(A_inv[:, coord_dim, ds_indexes], axis=1)
                    du = self.aether_element.eval_basis(quad_points, ds, d=d)            
                    yi = w[:,np.newaxis,np.newaxis] * du
                    y_d.append(yi)
                
                y_d = np.array(y_d).sum(axis=0)
                
            elif self.aether_element.ref_element == 'interval':
                # The 1d transformation case
                du = self.aether_element.eval_basis(quad_points, ds, d=d)
                w = (1. / self.mesh.edge_to_length)**len(ds)
                y_d = w[:,np.newaxis,np.newaxis] * du
                #print(yi.shape)
            
            Y.append(y_d)
            return Y 
        
        
    def get_cell_function(
        self,
        triangle_quad : TriangleQuadrature,
        interval_quad : IntervalQuadrature,
        derivatives = []
    ):
        
        element = self.element 
        if not element.ref_element == 'triangle':
            raise TypeError(f"The finite element must have a triangle reference element, but got {element.ref_element}.")
        
        """
        First, evaluate each basis function at quadrature points on each cell. 
        """
        quad_points = triangle_quad.quad_points
        Y = self.eval_basis(quad_points, derivatives)
        self.Y_cell = Y 
        
        """
        Then, evaluate each basis function at quadrature points on each edge. 
        """
        
        x = interval_quad.quad_points
        # Edge 0 
        x0 = np.c_[1. - x, x]
        Y0 = self.eval_basis(x0, derivatives)
        
        # Edge 1
        x1 = np.c_[0.*x, 1. - x]
        Y1 = self.eval_basis(x1, derivatives)
        
        # Edge 2 
        x2 = np.c_[x, 0.*x]
        Y2 = self.eval_basis(x2, derivatives)
        

class FunctionBuilder1:
      
    def __init__(self, 
        mesh : Mesh,
        interval_quad : IntervalQuadrature,
        triangle_quad : TriangleQuadrature
    ):
            
        """
        Object used to create finite element functions with basis functions evaluated on 
        particular quadratures. 

        Parameters
        ----------
        mesh : Mesh
            A mesh object.
            
        interval_quad : IntervalQuadrature
            The quadrature rule to use on 1d entities (edges).
            
        triangle_quad : TriangleQuadrature
            The quadrature rule to use on 2d entities (faces).
            
        """
        
        self.mesh = mesh
        self.interval_quad = interval_quad
        self.triangle_quad = triangle_quad 
        
    
    
    def get_function(self, element : Element, derivatives = [], entities = 'all'):
        
        """
        Returns a quadrature function for a set of quadrature points. Derivatives for each coordinate dimension
        can be passed in as additional arguments. 

        Parameters
        ----------
     
        derivatives : list of strings
            A list of derivatives to take for each basis function. For example, use eval_func(points, ['x', 'y']) to get the xy partial 
            derivatives. Can be left blank for no derivatives. 
            
        Returns
        -------
        QuadratureFunction
            A quadrature function used to evaluate a finite element function at quadrature points given DOF values. 
            
        """
        
        quad_points = quadrature.quad_points
        
        if self.aether_element.ref_element == 'triangle':
            indexes = [[0,1]]
        elif self.aether_element.ref_element == 'interval':
            indexes=[[0]]
        
        # Dimension of the range
        D = self.aether_element.range_dim
        # If derivatives aren't specified, just initialize empty lists for each range dimension
        if len(derivatives) == 0:
            derivatives = [[] for i in range(D)]

        symbol_dict = {'x' : 0, 'y' : 1}
        
        Y = []
        for d in range(D):
            
            ds = derivatives[d]
            ds_indexes = [symbol_dict[s] for s in ds]

            # Evaluate all basis functions on each mesh cell. 
            y_d = []
            
            """
            Evaluating derivatives in physical coordinates requires some somwehat unpleasant 
            chain ruling. See:
            https://scicomp.stackexchange.com/questions/25196/implementing-higher-order-derivatives-for-finite-element 
            """
            if self.aether_element.ref_element == 'triangle':
                
                A_inv = self.mesh.cell_to_A_inv
                for coord_dim in itertools.product(*(indexes*len(ds))):
                    # Weights are products of entries of the transform matrix
                    w = np.prod(A_inv[:, coord_dim, ds_indexes], axis=1)
                    du = self.aether_element.eval_basis(quad_points, ds, d=d)            
                    yi = w[:,np.newaxis,np.newaxis] * du
                    y_d.append(yi)
                
                y_d = np.array(y_d).sum(axis=0)
                
            elif self.aether_element.ref_element == 'interval':
                # The 1d transformation case
                du = self.aether_element.eval_basis(quad_points, ds, d=d)
                w = (1. / self.mesh.edge_to_length)**len(ds)
                y_d = w[:,np.newaxis,np.newaxis] * du
                #print(yi.shape)
            
            Y.append(y_d)
        
        """
        Combine any vector outputs into a single array. The resulting dimension of the array 
        is N x J x K x D. N is the number of cells for an element defined on a cell, or 
        the number of edges for an element defined on an interval. J is the number of of basis 
        functions per mesh entity (edge or cell). K is the number of quadrature points.  D is the 
        dimension of the range of each basis function (e.g. 1 for a scalar basis function or 2)
        for a vector basis function. 
        """
        Y = np.stack(Y, axis=-1)
        
        # For H(div) elements use a contravariant Piola transform
        if self.aether_element.continuity == 'H(div)':
            Y = self.contravariant_piola_transform(Y)
        elif self.aether_element.continuity == 'H(curl)':
            # For H(curl) elements use a covariant Piola transform
            Y = self.covariant_piola_transform(Y)
            
        # Extend quadrature to entire mesh 
        mesh_quad = MeshQuadrature(self.mesh, quadrature) 
    
        return Y, mesh_quad
        #f = QuadratureFunction(Y, mesh_quad, self)
        #return f
            
    
    def contravariant_piola_transform(self, y : NDArray):
        """
        Given an N x J x K x 2 array, perform a contravariant Piola transform.  
        """
        
        mesh = self.mesh 
        W = (mesh.cell_to_det_A)[:,np.newaxis,np.newaxis] * mesh.cell_to_A
        y = np.einsum('nij,nlkj->nlki', W, y)
        return y 
    
    
    def covariant_piola_transform(self, y : NDArray):
        """
        Given an N x J x K x 2 array, perform a covariant Piola transform.  
        """
        
        mesh = self.mesh 
        W = np.transpose(mesh.cell_to_A_inv, axes=(0,2,1))
        y = np.einsum('nij,nlkj->nlki', W, y)
        return y 
                
        
    def eval_basis(
        self, 
        entity_dim,
        entity_index,
        derivatives = []
    ):
        """
        Evaluates all finite element basis functions at a set of quadrature points.
        """
        
        if self.aether_element.ref_element == 'triangle':
            indexes = [[0,1]]
        elif self.aether_element.ref_element == 'interval':
            indexes=[[0]]
        
        # Dimension of the range
        D = self.aether_element.range_dim
        # If derivatives aren't specified, just initialize empty lists for each range dimension
        if len(derivatives) == 0:
            derivatives = [[] for i in range(D)]

        symbol_dict = {'x' : 0, 'y' : 1}
        
        Y = []
        for d in range(D):
            
            ds = derivatives[d]
            ds_indexes = [symbol_dict[s] for s in ds]

            # Evaluate all basis functions on each mesh cell. 
            y_d = []
            
            """
            Evaluating derivatives in physical coordinates requires some somwehat unpleasant 
            chain ruling. See:
            https://scicomp.stackexchange.com/questions/25196/implementing-higher-order-derivatives-for-finite-element 
            """
            if self.aether_element.ref_element == 'triangle':
                
                A_inv = self.mesh.cell_to_A_inv
                for coord_dim in itertools.product(*(indexes*len(ds))):
                    # Weights are products of entries of the transform matrix
                    w = np.prod(A_inv[:, coord_dim, ds_indexes], axis=1)
                    du = self.aether_element.eval_basis(quad_points, ds, d=d)            
                    yi = w[:,np.newaxis,np.newaxis] * du
                    y_d.append(yi)
                
                y_d = np.array(y_d).sum(axis=0)
                
            elif self.aether_element.ref_element == 'interval':
                # The 1d transformation case
                du = self.aether_element.eval_basis(quad_points, ds, d=d)
                w = (1. / self.mesh.edge_to_length)**len(ds)
                y_d = w[:,np.newaxis,np.newaxis] * du
                #print(yi.shape)
            
            Y.append(y_d)
            return Y 
        
        
    def get_cell_function(
        self,
        triangle_quad : TriangleQuadrature,
        interval_quad : IntervalQuadrature,
        derivatives = []
    ):
        
        element = self.element 
        if not element.ref_element == 'triangle':
            raise TypeError(f"The finite element must have a triangle reference element, but got {element.ref_element}.")
        
        """
        First, evaluate each basis function at quadrature points on each cell. 
        """
        quad_points = triangle_quad.quad_points
        Y = self.eval_basis(quad_points, derivatives)
        self.Y_cell = Y 
        
        """
        Then, evaluate each basis function at quadrature points on each edge. 
        """
        
        x = interval_quad.quad_points
        # Edge 0 
        x0 = np.c_[1. - x, x]
        Y0 = self.eval_basis(x0, derivatives)
        
        # Edge 1
        x1 = np.c_[0.*x, 1. - x]
        Y1 = self.eval_basis(x1, derivatives)
        
        # Edge 2 
        x2 = np.c_[x, 0.*x]
        Y2 = self.eval_basis(x2, derivatives)
        
    
    def get_edge_function(
        self,
        interval_quad : IntervalQuadrature,
        derivatives = []
    ):
        
        

        
        
        
        
        