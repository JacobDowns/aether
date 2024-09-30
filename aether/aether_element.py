import numpy as np
import symfem 
import sympy as sp
from sympy import lambdify

class AetherElement:
    
    def __init__(self, type, degree):
        """
        Implementation of an equidistant Lagrange finite element of arbitrary order. 

        Parameters
        ----------
        type : str 
            The type of finite element. 
        degree : int
            The degree of the finite element / polynomial degree of basis functions. 
            
        """
        
        if not type in ['Lagrange', 'discontinuous Taylor']:
            raise ValueError("Invlaid element type.")
        
        # The symfem definition of the element
        self.element = symfem.create_element('triangle', type, degree)
        
        self.basis_functions = self.element.get_basis_functions()
        
        dof_dims, dof_entities = list(zip(*self.element.dof_entities()))
        self.dof_dims = np.array(dof_dims, dtype=int)
        self.dof_entities = np.array(dof_entities, dtype=int)
        
        self.dofs_per_vertex = int(np.sum(self.dof_dims == 0) / 3)
        self.dofs_per_edge = int(np.sum(self.dof_dims == 1) / 3)
        self.dofs_per_face = int(np.sum(self.dof_dims == 2))
        
        # Number of basis functions
        self.N = len(self.basis_functions)
        
        # Plot positions of dofs on each entity
        vertex_dof_positions = []
        edge_dof_positions = []
        face_dof_positions = []

        for i in range(len(self.element.dof_entities())):
            dim, entity = self.element.dof_entities()[i]
            dof_x = self.element.dof_plot_positions()[i]
            dof_x = [float(dof_x[0]), float(dof_x[1])]
            
            if dim == 0 and entity == 0:
                vertex_dof_positions.append(dof_x)
            
            if dim == 1 and entity == 2:
                edge_dof_positions.append(dof_x)
                
            if dim == 2 and entity == 0:
                face_dof_positions.append(dof_x)
                
        self.vertex_dof_positions = np.array(vertex_dof_positions)
        self.edge_dof_positions = np.array(edge_dof_positions)
        self.face_dof_positions = np.array(face_dof_positions)

    
    def eval_func(self, points, *args):
        """
        Evaluates all basis function of the element at a set of points within the reference element which has 
        vertices (0,0), (1,0), (0,1).

        Parameters
        ----------
        points : ndarray
            A set of 2D points of shape n x 2 within the reference element. 
        
        
        Additional Arguments
        ----------
        d1, d2 ...
            Additional integer arguments. Each di must be 0 or 1. 0 denotes a derivative in the x direction, 1 in the y direction.
            For instance, to take the x y derivative of each basis function pass in eval_func(points, 0, 1).
            
        Returns
        -------
        ndarray
            All basis functions of the element evaluated at the given points. 
            
        """
        
        x, y = sp.symbols('x y')
        axes = list(args)
        
        coords = np.array([x, y])[axes]
        Z = []
        
        for i in range(self.N):
            f = self.basis_functions[i].as_sympy()
            if len(axes) > 0:
                f = sp.diff(f, *coords)
            f = lambdify((x, y), f)
            z = f(points[:,0], points[:,1])
            
            if isinstance(z, (int, float)):
                z = np.ones_like(points[:,0]) * z
                
            Z.append(z)
            
        Z = np.array(Z)
        return Z