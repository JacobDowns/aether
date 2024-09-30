import numpy as np 
import torch 
from aether.aether_mesh import Mesh 

class ReferenceQuadrature:
    
    def __init__(self, quad_points, quad_weights):
        """
        A quadrature rule on the reference triangle. 

        Parameters
        ----------
        quad_points : ndarray
            An n x 2 array, where n represents the number of quadrature points. 
            
        quad_weights: ndarray
            An array of length n containing the quadrature weights. 
      
        """
        
        self.quad_points = quad_points
        self.quad_weights = quad_weights
        
        
class MeshQuadrature:
    
    def __init__(self, mesh : Mesh, quadrature : ReferenceQuadrature):
        """
        Extends a quadrature rule on the reference triangle to the entire mesh.

        Parameters
        ----------
        mesh : Mesh
            A 2D mesh object. 
            
        quadrature: ReferenceQuadrature
            A quadrature rule defined on the reference element. 
      
        """
        
        self.mesh = mesh 
        self.quadrature = quadrature 
        
        # Compute quadrature points in mesh coordinates on a per cell basis
        reference_points = quadrature.quad_points
        quad_points = np.matmul(self.mesh.A, reference_points.T) + self.mesh.B[:,:,np.newaxis]
        mesh_quad_points = np.stack([quad_points[:,0,:], quad_points[:,1,:]], axis=2)
        self.mesh_quad_points = mesh_quad_points 
        
        # Just copy over the quad weights for convenience 
        self.quad_weights = self.quadrature.quad_weights 