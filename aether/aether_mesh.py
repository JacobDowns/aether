import numpy as np
from numpy.typing import NDArray
import torch

class Mesh:
    
    def __init__(self, coordinates : NDArray, cells : NDArray):
        """
        A simple mesh function that takes in a set of 2D vertex locations and an array of faces.

        Parameters
        ----------
        coordinates : ndarray
            An n x 2 array, where n represents the number of vertices, of vertex coordinates where each row
            represents an (x,y) coordinate of a given vertex.
            
        cells: ndarray
            An integer array of shape k x 3, where k is the number of triangular elements, and each row references
            3 vertex indices in the coordinates array. 
      
        """
        
        self.coordinates = coordinates
        self.cell_to_vertices = cells
        self.get_edges()
        self.set_cell_to_edge_maps()
        self.set_cell_transforms()
        self.set_edge_properties()
        self.set_cell_properties()
        self.set_edge_to_cells_map()
        
        self.num_vertices = len(self.coordinates)
        self.num_cells = len(self.cell_to_vertices)
        self.num_edges = len(self.edge_to_vertices)
        
        
    def get_edges(self):
        
        cells = self.cell_to_vertices
        
        """"
        Get all unique edges, and create a map from each edge index
        to its two vertices in sorted (lower to higher vertex index) order.
        """
        e0 = cells[:,[1,2]]
        e1 = cells[:,[0,2]]
        e2 = cells[:,[0,1]]
        
        e0.sort(axis=1)
        e1.sort(axis=1)
        e2.sort(axis=1)
        edges = np.concatenate((e0, e1, e2)) 
        edges = np.unique(edges, axis=0)
        self.edge_to_vertices = edges
        
    
    def set_cell_to_edge_maps(self):
        """
        Create maps from each cell to its three corresponding edge indexes. Additionally, create 
        a map that specifies the edge orientation within the cell. That is, does the edge go from 
        a lower to higher or higher to lower global index? 
        """
        
        cells = self.cell_to_vertices  
        edges = self.edge_to_vertices 
        
        """
        For consistency we consider the default orientation of an edge to be from the lower to higher 
        vertex index, but order in the array of cells is arbitrary. Hence, we keep track of orientation 
        for each edge (lower to higher, or higher to lower).
        """
        e0 = cells[:,[1,2]]
        e1 = cells[:,[0,2]]
        e2 = cells[:,[0,1]]
        
        orientation0 = e0[:,0] <= e0[:,1]
        orientation1 = e1[:,0] <= e1[:,1]
        orientation2 = e2[:,0] <= e2[:,1]
        self.cell_to_edges_orientation = np.array(
            np.c_[orientation0, orientation1, orientation2],
            dtype=int
        )
        
        # Create a map from each cell to corresponding edge indexes
        cell_to_edges = np.zeros((len(cells), 3), dtype=int)
        
        edge_dict = {}
        for i in range(len(edges)):
            edge_dict[tuple(edges[i])] = i 
            
        for i in range(len(cells)):
            cell = cells[i]
            
            i0 = min(cell[0], cell[1])
            i1 = max(cell[0], cell[1])
            edge = edge_dict[(i0,i1)]
            cell_to_edges[i,0] = edge 
            
            i0 = min(cell[1], cell[2])
            i1 = max(cell[1], cell[2])
            edge = edge_dict[(i0,i1)]
            cell_to_edges[i,1] = edge 
            
            i0 = min(cell[0], cell[2])
            i1 = max(cell[0], cell[2])
            edge = edge_dict[(i0,i1)]
            cell_to_edges[i,2] = edge 
            
        self.cell_to_edges = cell_to_edges
        
        
    def set_edge_properties(self):
        """
        Creates maps from edges to edge properties including normal vectors, tangent vectors,
        lengths, and midpoints. 
        """
        
        edge_to_vertices = self.edge_to_vertices
        coordinates = self.coordinates
        
        # Tangent vectors
        t = coordinates[edge_to_vertices[:,0]] - coordinates[edge_to_vertices[:,1]]
        # Edge lengths
        edge_lens = np.linalg.norm(t, axis=1)
        t /= edge_lens[:,np.newaxis]
        # Normal vectors 
        n = np.stack([t[:,1], -t[:,0]], axis=-1)
        # Edge midpoint coordinates
        edge_midpoints =  0.5*(coordinates[edge_to_vertices[:,0]] + coordinates[edge_to_vertices[:,1]])
        
        self.edge_to_length = edge_lens 
        self.edge_to_midpoint = edge_midpoints
        self.edge_to_tangent = t 
        self.edge_to_normal = n 
        
        
    def set_cell_transforms(self):
        """
        Create affine maps from reference to physical elements and vice versa. 
        """

        cells = self.cell_to_vertices 
        coordinates = self.coordinates
        
        X = coordinates[:,0][cells]
        Y = coordinates[:,1][cells]
        
        # Offsets for affine transform
        B = np.c_[X[:,0], Y[:,0]]

        # Transformation matrices from triangle reference element
        A = np.zeros((len(cells), 2, 2))
        A[:,0,0] = X[:,1] - X[:,0]
        A[:,0,1] = X[:,2] - X[:,0]
        A[:,1,0] = Y[:,1] - Y[:,0]
        A[:,1,1] = Y[:,2] - Y[:,0]

        # Determinants of transformation matrices
        det_A = A[:,0,0]*A[:,1,1] - A[:,0,1]*A[:,1,0]

        # Inverse transform to reference element
        A_inv = np.zeros_like(A)
        A_inv[:,0,0] = A[:,1,1]
        A_inv[:,0,1] = -A[:,0,1]
        A_inv[:,1,0] = -A[:,1,0]
        A_inv[:,1,1] = A[:,0,0]
        A_inv = (1. / det_A[:,np.newaxis,np.newaxis]) * A_inv 
        
        # Maps from cells to their transformation matrices and offsets
        self.cell_to_A = A 
        self.cell_to_A_inv = A_inv
        self.cell_to_det_A = det_A
        self.cell_to_B = B
        
        
    def set_cell_properties(self):
        """
        Sets maps from each cell to properties like its area, the length of its edges, tangent 
        vectors of its edges, outward pointing normals for each of its edges, area, and centroids.
        Note that the normal vector and tangent vectors for an edge are not necessarily the same 
        as in the edge maps. 
        """
        
        coordinates = self.coordinates
        cells = self.cell_to_vertices 
        
        # Tangent vectors
        t = coordinates[cells[:,[1,2,0]]] - coordinates[cells[:,[0,1,2]]]
        # Edge lengths
        edge_lens = np.linalg.norm(t, axis=2)
        # Normalize
        t = t / edge_lens[:,:,np.newaxis]
        # Edge midpoint coordinates
        edge_midpoints =  0.5*(coordinates[cells[:,[1,2,0]]] + coordinates[cells[:,[0,1,2]]])
        # Outward pointing normal vectors 
        n = np.stack([t[:,:,1], -t[:,:,0]], axis=-1)
        # Centroid
        centroids = coordinates[cells].sum(axis=1) / 3.
        
        self.cell_to_edge_tangents = t 
        self.cell_to_edge_normals = n
        self.cell_to_edge_lens = edge_lens
        self.cell_to_edge_midpoints = edge_midpoints
        self.cell_to_area = np.absolute(self.cell_to_det_A) / 2.
        self.cell_to_centroid = centroids
        
    
    def set_edge_to_cells_map(self):
        """
        Create maps from each edge to its one or two associated cells. Identify interior and exterior
        edges. Identify the +/- side of the edge. If the edge normal is the opposite of the outward normal 
        of the edge in a cell, that cell is on the + side. Otherwise it's on the - side. 
        """
        
        edge_to_cells_map = np.zeros((len(self.edge_to_vertices), 2), dtype=int) -1
        edge_to_cell_edges = np.zeros((len(self.edge_to_vertices), 2), dtype=int) -1
        
        for i in range(len(self.cell_to_edges)):
            for j in range(len(self.cell_to_edges[i])):
                e = self.cell_to_edges[i,j]
                # Get the normal vector of this edge 
                n_edge = self.edge_to_normal[e]
                # Get the outward normal of this cell edge
                n_cell = self.cell_to_edge_normals[i,j]
               
                if np.dot(n_edge, n_cell) > 0.:
                    edge_to_cells_map[e,1] = i 
                    edge_to_cell_edges[e,1] = (j+2)%3
                else:
                    edge_to_cells_map[e,0] = i
                    edge_to_cell_edges[e,0] = (j+2)%3
                    
        self.edge_to_cells = edge_to_cells_map
        self.edge_to_cell_edges = edge_to_cell_edges
        
        # Indexes of interior and exterior edges
        indexes = np.logical_and(edge_to_cells_map[:,0] >= 0, edge_to_cells_map[:,1] >= 0)
        interior_edges = np.argwhere(indexes).flatten()
        exterior_edges = np.argwhere(~indexes).flatten()
        self.interior_edges = interior_edges
        self.exterior_edges = exterior_edges
        
        # Interior edges to two adjacent cells
        self.interior_edge_to_cells = self.edge_to_cells[interior_edges]
        self.interior_edge_to_cell_edges = self.edge_to_cell_edges[interior_edges]
        
        # Exterior edge to one adjacent cell
        self.exterior_edge_to_cell = self.edge_to_cells[exterior_edges].max(axis=1)
        self.exterior_edge_to_cell_edge = self.edge_to_cell_edges[exterior_edges].max(axis=1)

       
    def cell_transform(self, x : NDArray):
        """
        Given a an nx2 array of points in the reference cell, generate all of the corresponding
        points on each mesh cell. 
        """    
        
        A = self.cell_to_A 
        b = self.cell_to_B
        y = np.matmul(A, x.T) + b[:,:,np.newaxis]
        y = np.transpose(y, axes=(0,2,1))
        
        return y 
    
    def edge_transform(self, x : NDArray):
        """
        Given a an nx1 array of points in the reference interval, generate all of the corersponding
        points on each edge.
        """    
        
        coords0 = self.coordinates[self.edge_to_vertices[:,0]]
        coords1 = self.coordinates[self.edge_to_vertices[:,1]]
        x = x[np.newaxis,:,np.newaxis]
        y = coords1[:,np.newaxis,:]*x + coords0[:,np.newaxis,:]*(1.-x)
        return y
    
    def contravariant_piola_transform(self, x : NDArray):
        W = (1. / self.cell_to_det_A)[:,np.newaxis,np.newaxis] * self.cell_to_A 
        y = np.einsum('nij,nklj->nklj', W, y)
        return y 