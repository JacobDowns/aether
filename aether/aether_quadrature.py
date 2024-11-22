import numpy as np 
from aether.aether_mesh import Mesh 
from numpy.typing import NDArray


class ReferenceQuadrature:
    
    def __init__(
        self,
        quad_points : NDArray,
        quad_weights : NDArray
    ):  
        """
        Generic quadrature class. 
        """
        
        self.quad_points = quad_points
        self.quad_weights = quad_weights


class TriangleQuadrature(ReferenceQuadrature):
    
    def __init__(
            self, 
            quad_points : NDArray,
            quad_weights : NDArray,
        ):
        
        """
        A quadrature rule on the reference triangle. 

        Parameters
        ----------
        quad_points : ndarray
            An nx2 array for the reference triangle or length n array for 
            the reference triangle.
            
        quad_weights: ndarray
            An array of length n containing the quadrature weights. 
        """
        
        ReferenceQuadrature.__init__(self, quad_points, quad_weights)
      

class IntervalQuadrature:
    
    def __init__(
            self, 
            quad_points : NDArray,
            quad_weights : NDArray,
        ):
        
        """
        A quadrature rule on the reference interval. 

        Parameters
        ----------
        quad_points : ndarray
            A length n array for the reference triangle or length n array for 
            the reference interval.  
            
        quad_weights: ndarray
            An array of length n containing the quadrature weights. 
        """
        
        ReferenceQuadrature.__init__(self, quad_points, quad_weights)
        
        
class MeshQuadrature:
    
    def __init__(self, mesh : Mesh, quadrature : ReferenceQuadrature):
        """
        Extends a quadrature rule on the reference triangle to the entire mesh.

        Parameters
        ----------
        mesh : Mesh
            A 2D mesh object. 
            
        quadrature: ReferenceQuadrature
            A quadrature rule defined on the reference triangle or interval.
      
        """
        
        self.mesh = mesh 
        self.quadrature = quadrature 
        reference_points = quadrature.quad_points
        
        if isinstance(quadrature, TriangleQuadrature):
            mesh_quad_points = mesh.cell_transform(reference_points)
            #mesh_quad_points = np.transpose(mesh_quad_points, axes=(0,2,1))
            #print(mesh_quad_points.shape)
        elif isinstance(quadrature, IntervalQuadrature):
            mesh_quad_points = mesh.edge_transform(reference_points)
        
        # All quadrature points on the mesh 
        self.quad_points = mesh_quad_points  
        # Just copy over the quad weights for convenience 
        self.quad_weights = self.quadrature.quad_weights 