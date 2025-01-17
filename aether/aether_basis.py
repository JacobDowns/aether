from __future__ import annotations
import torch
from jaxtyping import Float
      
class QuadratureBasis:
     def __init__(
        self, 
        quad_points :  Float[torch.tensor, 'entity sub_entity quad_point dim'],
        quad_weights : Float[torch.tensor, 'quad_points'],
        basis_vals : Float[torch.tensor, 'entity subentity basis_func quad_points range_dim'],
        entity_dim = 1,
        derivatives = []
    ):
        
        self.size = basis_vals.shape[2]
        self.quad_points = quad_points 
        self.quad_weights = quad_weights
        self.basis_vals = basis_vals
        self.entity_dim = entity_dim 
        self.derivatives = derivatives
      