import numpy as np 
from aether.aether_mesh import Mesh 
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from aether.aether_quadrature import Quadrature, PointQuadrature, IntervalQuadrature, TriangleQuadrature


class ReferenceElement(ABC):

    @abstractmethod
    def map_quadrature_to_entity(self, quadrature : Quadrature, entity_dim = 0, entity_index = 0):
        """
        Reference elements consist of subentities of different dimensions including entities of dimension 3
        (tetrahedra), entities of dimension 2 (triangles), entities of dimension 1 (intervals), and entities
        of dimension 0 (points). Sometimes we want to evaluate finite element basis functions on particular 
        subentities of a reference element. Given generic quadrature rules, this function maps the quadrature points
        to the subentity of interest. 
        
        Parameters
        ----------
        quadrature : Quadrature
            A quadrature whose type is appropriate for the given dimension entity. 
            
        entity_dim : int
            The dimension of the target entity (2 for face, 1 for edge, 0 for point).
            
        entity_index : int
            The index of the target entity. For example, 0 for the first entity of a given 
            dimension.      
            
        """
        
        if entity_dim == 2:
            if not isinstance(quadrature, TriangleQuadrature):
                raise ValueError(
                f"For an entity of dimension 2, quadrature must be a TriangleQuadrature."
                f"Got type {type(quadrature)}."
            ) 
        elif entity_dim == 1:
            if not isinstance(quadrature, IntervalQuadrature):
                raise ValueError(
                f"For an entity of dimension 1, quadrature must be an IntervalQuadrature."
                f"Got type {type(quadrature)}."
            )
        elif entity_dim == 0:
            if not isinstance(quadrature, PointQuadrature):
                raise ValueError(
                f"For an entity of dimension 0, quadrature must be a PointQuadrature."
                f"Got type {type(quadrature)}."
            )
        else :
            raise ValueError(
                f"Unsupported entity dimension."
                f"Got dimension {entity_dim}."
            )
    
    

class ReferenceTriangle(ReferenceElement):
    
    def __init__(
        self,
    ):  
        """
        Defines the reference triangle. 
        """
        
        self.vertices = np.array([
            [0.,0.],
            [1.,0.],
            [0.,1.]
        ])
        
        self.edges = np.array([
            [1, 2],
            [2, 0],
            [0, 1]
        ])
        
        self.faces = np.array([
            [0,1,2]
        ])
        
        self.entities = {
            0 : [0, 1, 2],
            1 : [0, 1, 2],
            2 : [0],
            3 : []
        }
        
        self.num_vertices = 3
        self.num_edges = 3
        self.num_faces = 1
        self.num_tetrahedra = 0
        self.dimension = 2
        
    
    def map_quadrature_to_entity(self, quadrature : Quadrature, entity_dim = 2, entity_index = 0) -> TriangleQuadrature:
        """
        Map quadrature rule to a geometric subentity of the reference element. 
        """
        
        super().map_quadrature_to_entity(quadrature, entity_dim, entity_index)
        
        if entity_dim == 2:
            
            if entity_index == 0:
                x = quadrature.quad_points
            else:
                raise ValueError(
                    f"The reference triangle only has one face, so entity_index must be 0."
                    f"Got {entity_index}."
                )

        elif entity_dim == 1:
            
            if entity_index in set([0,1,2]):
                t = quadrature.quad_points[:, np.newaxis]
                edge = self.edges[entity_index]
                x0 = self.vertices[edge[0]][np.newaxis, :]
                x1 = self.vertices[edge[1]][np.newaxis, :]
                
                x = x0*(1-t) + t*x1 
            else :
                raise ValueError(
                    f"The reference triangle only has three edges, so entity_index must be 0, 1, or 2."
                    f"Got {entity_index}."
                )
            
        elif entity_dim == 0:
            
            if entity_index in set([0,1,2]):
                x = np.array([self.vertices[entity_index]])
            else:
                raise ValueError(
                    f"The reference triangle only has three vertices, so entity_index must be 0, 1, or 2."
                    f"Got {entity_index}."
                )
        
        # Return a triangle quadrature        
        quad = TriangleQuadrature(x, quadrature.quad_weights)
        return quad 
    
    
class ReferenceInterval(ReferenceElement):
    
    def __init__(self):
        """
        Defines the reference interval (0, 1). 
        """
        
        self.vertices = [0, 1]
        self.edges = [0]   
        
        
        self.entities = {
            0 : [0, 1],
            1 : [0],
            2 : [],
            3 : []
        }
        
        self.num_vertices = 2
        self.num_edges = 1
        self.num_faces = 0
        self.num_tetrahedra = 0
        self.dimension = 1
        
        
    def map_quadrature_to_entity(self, quadrature : Quadrature, entity_dim = 1, entity_index = 0):
        """
        Map quadrature rule to a geometric subentity of the reference element. 
        """
        
        super().map_quadrature_to_entity(quadrature, entity_dim, entity_index)
        
        if entity_dim == 1:
            x = quadrature.quad_points 
        elif entity_dim == 0:
            if entity_index in set([0,1]):
                x = np.array([self.vertices[entity_index]])
            else:
                 raise ValueError(
                    f"The reference interval only has two vertices, so entity_index must be 0 or 1."
                    f"Got {entity_index}."
                )
        
        # Return an interval quadrature      
        quad = IntervalQuadrature(x, quadrature.quad_weights)
        return quad 
    