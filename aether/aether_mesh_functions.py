import numpy as np
from numpy.typing import NDArray
from aether.aether_mesh import Mesh


class CellFunction:
    """
    A function associated with subentities of a cell. 
    """
    
    def __init__(self, mesh : Mesh, entity_dim=2, n=1):
        self.mesh = mesh 
        self.entity_dim = entity_dim 
        self.n = n 
        
        if self.entity_dim == 2:
            self.x = np.zeros((mesh.num_cells, 1, n))
        elif self.entity_dim == 1:
            self.x = np.zeros((mesh.num_cells, 3, n)) 
        elif self.entity_dim == 0:
            self.x = np.zeros((mesh.num_cells, 1, n))
        
        
        
        
       