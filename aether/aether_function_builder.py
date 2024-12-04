import numpy as np
import itertools
from aether.aether_element import Element
from aether.aether_mesh import Mesh
from aether.aether_quadrature import TriangleQuadrature, IntervalQuadrature, PointQuadrature, MeshQuadrature
from aether.aether_functions import CellFunction, EdgeFunction
from numpy.typing import NDArray
import torch 

class FunctionBuilder:
      
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
        self.point_quad = PointQuadrature()
        
      
    def eval_basis(
        self, 
        element : Element,
        entity_dim : 2,
        entity_index : 0,
        derivatives = [], 
        device='cpu'
    ):
        """
        Evaluates all finite element basis functions at a set of quadrature points.
        """
        
        if element.ref_element_name == 'triangle':
            indexes = [[0,1]]
        elif element.ref_element_name == 'interval':
            indexes=[[0]]
        
        # Dimension of the range
        D = element.range_dim
        
        # If derivatives aren't specified, just initialize empty lists for each range dimension
        if len(derivatives) == 0:
            derivatives = [[] for i in range(D)]

        symbol_dict = {'x' : 0, 'y' : 1}
        symbols = ['x', 'y']
        
        # Use the appropriate quadrature point for the dimension
        if entity_dim == 0:
            quad = self.point_quad 
        elif entity_dim == 1:
            quad = self.interval_quad
        elif entity_dim == 2:
            quad = self.triangle_quad
        
        ref_element = element.ref_element 
        quad_points = quad.quad_points
        q = ref_element.map_quadrature_to_entity(quad, entity_dim, entity_index)
        quad_points = q.quad_points 
        
        Y = []
        for d in range(D):
            
            ds = derivatives[d]
            ds_indexes = [symbol_dict[s] for s in ds]

            # Evaluate all basis functions on each mesh cell. 
            y_d = []
            
            """
            Evaluating derivatives in physical coordinates requires some somwehat unpleasant 
            chain ruling. Basically, derivatives in physical coordinates are weighted sums of 
            derivatives in reference coordinates. The weights are given by products of entries 
            in the transform Jacobian matrix. See:
            https://scicomp.stackexchange.com/questions/25196/implementing-higher-order-derivatives-for-finite-element 
            """
            if element.ref_element_name == 'triangle':
                

                A_inv = self.mesh.cell_to_A_inv
                for coord_dim in itertools.product(*(indexes*len(ds))):
                    derivative = [symbols[k] for k in coord_dim]
                    w = np.prod(A_inv[:, coord_dim, ds_indexes], axis=1)
                    du = element.eval_basis(quad_points, derivative, d=d)            
                    yi = w[:,np.newaxis,np.newaxis] * du
                    y_d.append(yi)
                
                y_d = np.array(y_d).sum(axis=0)
                
            elif element.ref_element_name == 'interval':
                # The 1d transformation case
                
                du = element.eval_basis(quad_points, ds, d=d)
                w = (1. / self.mesh.edge_to_length)**len(ds)
                y_d = w[:,np.newaxis,np.newaxis] * du
            
            Y.append(y_d)
        
        # Stack dimensions
        Y = np.stack(Y, axis=-1)
        
        # Apply appropriate transformations for vector elements
        if element.continuity == 'H(div)':
            Y = self.contravariant_piola_transform(Y)
        elif element.continuity == 'H(curl)':
            Y = self.covariant_piola_transform(Y)
        
        # Extend the quadrature rule to the entire mesh
        mesh_quad = MeshQuadrature(self.mesh, q, device=device)
        return Y, mesh_quad 
        
    
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
    
    
    def create_function(self, element : Element, entity_dims=[1,2], derivatives = [], device='cuda'):
        
        bases = {}
        quadratures = {}
        
        for entity_dim in entity_dims:
            bases[entity_dim] = []
            quadratures[entity_dim] = []
            for entity_index in element.ref_element.entities[entity_dim]:
                Y, mesh_quad = self.eval_basis(element, entity_dim, entity_index, derivatives)
                Y = torch.tensor(Y, dtype=torch.float32, device=device)
                bases[entity_dim].append(Y)
                quadratures[entity_dim].append(mesh_quad)
                    
        if element.ref_element_name == 'triangle':
            f = CellFunction(self.mesh, element, bases, quadratures, device)              
        elif element.ref_element_name == 'interval':
            f = EdgeFunction(self.mesh, element, bases, quadratures, device)        
       
        return f