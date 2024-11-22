import numpy as np
from aether.aether_element import Element
from aether.aether_mesh import Mesh
from aether.aether_quadrature import IntervalQuadrature

"""
Implements common degrees of freedom used in finite elements. Degrees of freedom
are functionals that map the basis functions to a scalar value. 
"""

class EdgeNormalDof:
    
    def __init__(
        self, 
        mesh : Mesh,
        element : Element,
        quadrature : IntervalQuadrature,
        weight_func = None
    ):
        self.element = element 
        
        # Evaluate the basis function on each edge 
        t = quadrature.quad_points 
        
        
        element.eval_basis()
        
        
class EdgeIntegral:
    
    def __init__(
        self,
        mesh : Mesh 
    )