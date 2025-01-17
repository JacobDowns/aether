import numpy as np 
from aether.aether_mesh import Mesh 
from numpy.typing import NDArray
from abc import ABC
from basix import CellType
import basix
import torch 

class Quadrature:
    
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
        
        
class PointQuadrature(Quadrature):
    
    def __init__(
        self
    ):
        """
        Quadrature on a a point (entity of dimension 0). 
        Yeah, this is pretty stupid, but don't judge me bro. 
        """
        
        self.quad_points = np.array([0.])
        self.quad_weights = np.array([1.])
        

class IntervalQuadrature(Quadrature):
    
    def __init__(
            self, 
            quad_points : NDArray,
            quad_weights : NDArray,
        ):
        
        """
        A quadrature rule on the reference interval from 0 to 1 (entity of dimension 1).

        Parameters
        ----------
        quad_points : ndarray
            A length n array of points on the unit interval. 
            
        quad_weights: ndarray
            An array of length n containing the quadrature weights. 
        """
        
        # Size check
        if quad_points.ndim != 1:
            raise ValueError(
                f"Expected quadrature points to be a 1d array. "
                f"Got shape {quad_points.shape}."
            )
        
        if quad_weights.ndim != 1 or quad_points.shape[0] != quad_weights.shape[0]:
            raise ValueError(
                f"Expected quadrature weights to be 1D array with same length as quadrature points."
                f"Got shape {quad_points.shape}."
            )
            
        # Make sure quadrature points are in reference interval.
 
        if quad_points.min() < 0. or quad_points.max() > 1.:
            raise ValueError(
                f"Quadrature points must be within the unit interval (0,1)."
            )
        
        Quadrature.__init__(self, quad_points, quad_weights)
        

class TriangleQuadrature(Quadrature):
    
    def __init__(
            self, 
            quad_points : NDArray,
            quad_weights : NDArray,
        ):
        
        """
        A quadrature rule on the reference triangle (entity of dimension 2). 

        Parameters
        ----------
        quad_points : ndarray
            An nx2 array of points within the reference triangle. 
            
        quad_weights: ndarray
            An array of length n containing the quadrature weights. 
            
        """
        
        # Size check
        if quad_points.ndim != 2 or quad_points.shape[1] != 2:
            raise ValueError(
                f"Expected quadrature points to have shape (n, 2). "
                f"Got shape {quad_points.shape}."
            )
        
        if quad_weights.ndim != 1 or quad_points.shape[0] != quad_weights.shape[0]:
            raise ValueError(
                f"Expected quadrature weights to be 1D array with same length as quadrature points."
                f"Got shape {quad_points.shape}."
            )
            
        # Make sure quadrature points are in reference triangle.
        x = quad_points[:,0]
        y = quad_points[:,1]
        if x.min() < 0. or y.min() < 0. or np.any(y > 1.-x + 1e-5):
            raise ValueError(
                f"Quadrature points must be within reference triangle with vertices (0,0), (1,0), (0,1)."
            )
        
        Quadrature.__init__(self, quad_points, quad_weights)


class BasixTriangleQuadrature(TriangleQuadrature):
    """
    Convenience wrapper for a Basix quadrature rule on a triangle of a particular degree. 
    """    
    
    def __init__(self, degree):
        self.degree = degree 
        quad_points, quad_weights = basix.make_quadrature(CellType.triangle, degree)
        super().__init__(quad_points, quad_weights)
        


class BasixIntervalQuadrature(IntervalQuadrature):
    """
    Convenience wrapper for a Basix quadrature rule on an interval of a particular degree. 
    """    
    
    def __init__(self, degree):
        self.degree = degree 
        quad_points, quad_weights = basix.make_quadrature(CellType.interval, degree)
        super().__init__(quad_points.flatten(), quad_weights)
        
        
class MeshQuadrature:
    
    def __init__(self, mesh : Mesh, quadrature : Quadrature, device='cuda'):
        """
        Extends a quadrature rule on a reference element to the entire mesh. 

        Parameters
        ----------
        mesh : Mesh
            A 2D mesh object. 
            
        quadrature: Quadrature
            A quadrature rule defined on the reference triangle or interval.
      
        """
        
        self.mesh = mesh 
        self.quadrature = quadrature 
        reference_points = quadrature.quad_points
        
        if isinstance(quadrature, TriangleQuadrature):
            mesh_quad_points = mesh.cell_transform(reference_points)
            #mesh_quad_points = np.transpose(mesh_quad_points, axes=(0,2,1))
        elif isinstance(quadrature, IntervalQuadrature):
            mesh_quad_points = mesh.edge_transform(reference_points)
        
        # All quadrature points on the mesh 
        self.quad_points = mesh_quad_points 
        # Just copy over the quad weights for convenience 
        self.quad_weights = quadrature.quad_weights