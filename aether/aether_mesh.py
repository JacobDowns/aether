import numpy as np

class Mesh:
    
    def __init__(self, coordinates, faces):
        """
        A simple mesh function that takes in a set of 2D vertex locations and an array of faces.

        Parameters
        ----------
        coordinates : ndarray
            An n x 2 array, where n represents the number of vertices, of vertex coordinates where each row
            represents an (x,y) coordinate of a given vertex.
            
        faces: ndarray
            An integer array of shape k x 3, where k is the number of triangular elements, and each row references
            3 vertex indices in the coordinates array. 
      
        """
        
        
        self.coordinates = coordinates
        self.faces = faces
        
        # Map from edges to vertices
        e0 = self.faces[:,[1,2]]
        e1 = self.faces[:,[0,2]]
        e2 = self.faces[:,[0,1]]

        # For consistenncy we considered the default orientation of an edge to be from the lower to higher 
        # vertex index, but this isn't enforced. Instead we keep track of orientation (lower to higher, or higher to lower)
        orientation0 = e0[:,0] <= e0[:,1]
        orientation1 = e1[:,0] <= e1[:,1]
        orientation2 = e2[:,0] <= e2[:,1]
        self.faces_to_edge_orientation = np.array(np.c_[orientation0, orientation1, orientation2], dtype=int)
        
        # Get an array of edges that reference to vertex indexes
        e0.sort(axis=1)
        e1.sort(axis=1)
        e2.sort(axis=1)
        edges = np.concatenate((e0, e1, e2)) 
        edges = np.unique(edges, axis=0)
        self.edges_to_vertices = edges
        
        # Create a map from faces to edges. That is, which three edges in the edges array form each face.
        edge_dict = {tuple(edges[i]) : i for i in range(len(edges))}
        
        face_to_edges = np.zeros((len(faces), 3), dtype=int)
        for i in range(len(faces)):
            face = faces[i]
            
            i0 = min(face[0], face[1])
            i1 = max(face[0], face[1])
            edge = edge_dict[(i0,i1)]
            face_to_edges[i,0] = edge 
            
            i0 = min(face[1], face[2])
            i1 = max(face[1], face[2])
            edge = edge_dict[(i0,i1)]
            face_to_edges[i,1] = edge 
            
            i0 = min(face[0], face[2])
            i1 = max(face[0], face[2])
            edge = edge_dict[(i0,i1)]
            face_to_edges[i,2] = edge 
        
        self.faces_to_edges = face_to_edges
        self.num_faces = len(faces)
        self.num_edges = len(edges)
        self.num_vertices = len(self.coordinates)
        
        # Compute per cell transformation matrices from the reference element
        # to each physical element as well as inverse transforms, and determinants for 
        # integration
        self.set_cell_transforms()
        
    def set_cell_transforms(self):
        
        faces = self.faces
        faces = faces[:,[0,1,2]]
        X = self.coordinates[:,0][faces]
        Y = self.coordinates[:,1][faces]

        B = np.c_[X[:,0], Y[:,0]]
        A = np.zeros((len(faces), 2, 2))

        # Transformation matrices from triangle reference element
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
        
        self.A = A 
        self.A_inv = A_inv
        self.det_A = det_A
        self.B = B
 