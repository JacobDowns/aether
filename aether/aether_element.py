import numpy as np
import symfem 
import sympy as sp
from sympy import lambdify
from numpy.typing import NDArray
from aether.aether_reference_elements import ReferenceTriangle, ReferenceInterval

class Element:
    
    def __init__(self, type : str, degree : int, ref_element_name='triangle'):
        
        # Type of reference element 
        self.ref_element_name = ref_element_name 
        # The symfem definition of the element
        self.element = symfem.create_element(ref_element_name, type, degree)
        # Get basis function 
        self.basis_functions = self.element.get_basis_functions()
        # Number of basis functions
        self.num_basis_functions = len(self.basis_functions)
        # Number of components of basis functions (scalar 1 v. vector 2)
        self.range_dim = self.element.range_dim
        # Continuity of the element (e.g. C0, H(div), etc.)
        self.continuity = self.element.continuity
        
        dof_dims, dof_entities = list(zip(*self.element.dof_entities()))
        self.dof_dims = np.array(dof_dims, dtype=int)
        self.dof_entities = np.array(dof_entities, dtype=int)
        
        if ref_element_name == 'interval':
            self.ref_element = ReferenceInterval() 
        elif ref_element_name == 'triangle':
            self.ref_element = ReferenceTriangle()
        else:
            raise ValueError(
                f"Only elements on intervals or triangles are currently supported"
                f"Got element type {ref_element_name}."
            )
            
        self.dofs_per_vertex = int(np.sum(self.dof_dims == 0) / self.ref_element.num_vertices)
        self.dofs_per_edge = int(np.sum(self.dof_dims == 1) / self.ref_element.num_edges)
        if self.ref_element.num_faces > 0:
            self.dofs_per_face = int(np.sum(self.dof_dims == 2))
        else: 
            self.dofs_per_face = 0
        
        
        # For each entity of a given type (face, edge, vertex)
        # create a list of its dof plot vertices 
        vertex_dof_positions = [[] for i in range(self.ref_element.num_vertices)]
        edge_dof_positions = [[] for i in range(self.ref_element.num_edges)]
        face_dof_positions = [[] for i in range(self.ref_element.num_faces)]
        for i in range(len(self.element.dof_entities())):
            dim, entity = self.element.dof_entities()[i]
            dof_x = self.element.dof_plot_positions()[i]
            dof_x = [float(d_i) for d_i in dof_x]
            
            if dim == 0:
                vertex_dof_positions[entity].append(dof_x)
            if dim == 1:
                edge_dof_positions[entity].append(dof_x)
            if dim == 2:
                face_dof_positions[entity].append(dof_x)
        
        self.vertex_dof_positions = np.array(vertex_dof_positions)
        self.edge_dof_positions = np.array(edge_dof_positions)
        self.face_dof_positions = np.array(face_dof_positions)
                
     
    def eval_basis(self, points : NDArray, derivatives = [], d=0):
        """
        Evaluates all basis function of the finite element. 

        Parameters
        ----------
        points : ndarray
            A set of 2D points of shape n x 2 within the reference element. 
        derivatives : list of strings
            A list of derivatives to take for each basis function. For example, use eval_func(points, ['x', 'y']) to get the xy partial 
            derivatives. Can be left blank for no derivatives. 
        d : int
            Which component of the range to evaluate. 
      
        Returns
        -------
        ndarray
            All basis functions evaluated at the given points. The final shape is N x P, where N is the number of basis functions
            and P is the number of points. 
            
        """
                
        # Map strings to symbols for sympy
        x, y = sp.symbols('x y')
        symbol_dict = {'x' : x, 'y' : y}
        derivatives = [symbol_dict[s] for s in derivatives]
        
        # Interval and triangle elements have different domains so 
        # make sure we have the correct format for points
        num_points = len(points)
        if self.ref_element_name == 'triangle':
            points = [points[:,0], points[:,1]] 
            domain = (x, y)
        elif self.ref_element_name == 'interval':
            points = [points.flatten()]
            domain = (x,)
        
        # List to store all evaluated basis functions
        Z = []
 
        for i in range(self.num_basis_functions):
            # Get i-th basis function 
            f = self.basis_functions[i].as_sympy()
            
            if self.range_dim > 1:
                f = f[d]
            
            # Differentiate the basis function 
            if len(derivatives) > 0:
                f = sp.diff(f, *derivatives)
            
            # Convert from a sympy function to a numerical representation
            f = lambdify(tuple(domain), f)
            # Evaluate the basis function at a set of points
            z = f(*points)
            
            # Handle edge case where functions are constants
            if isinstance(z, (int, float)):
                z = np.ones(num_points) * z
            
            Z.append(z)
            
        Z = np.array(Z)
        return Z
