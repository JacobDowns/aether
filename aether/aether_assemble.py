import torch 
from itertools import combinations, product
import numpy as np 
import ast
import inspect

def assemble(f, basis_names, range_dim=0):
    
    N = len(basis_names)
    bases = [f.quad_bases[basis_names[i]] for i in range(N)]
    combs = []
    for basis in bases:
        combs.append(range(basis.size))
    
    indices = np.array(list(product(*combs))).T
    
    B = []
    for i in range(N):
        basis = bases[i]
        b_i = basis.basis_vals[:,:,indices[i],:,range_dim]
        B.append(b_i)
        w = basis.quad_weights

    dofs = f.local_to_global_dofs
    dofs = dofs[:,indices.T]
    
    # Compute integrals of products of basis functions
    mesh = f.mesh 
    det_A = torch.tensor(np.absolute(mesh.cell_to_det_A), dtype=torch.float32, device=f.device)
    B = torch.prod(torch.column_stack(B), dim=1)
    I = torch.sum(B*w[None, None, :], dim=-1)
    I = (I*det_A[:,None]).flatten()
    
    # Global dofs corresponding to products of basis functions
    dofs = dofs.reshape(-1,N).T
    
    M = torch.sparse_coo_tensor(dofs, I, device=f.device)
    M.coalesce()
    M = M.to_dense()
    
    print(M)
    quit()
    
    return M 
    #M = M.to


# Extract variables from a form 
def extract_variables(form):
    source = inspect.getsource(form)
    tree = ast.parse(source)
    variable = list(inspect.signature(form).parameters.keys())[0]
    print(variable)

    class VariableExtractor(ast.NodeVisitor):
        def __init__(self):
            self.variables = []

        def visit_Call(self, node):
            # Look for calls to u('x'), u('y'), etc.
            if isinstance(node.func, ast.Name) and node.func.id == variable:
                if node.args and isinstance(node.args[0], ast.Constant):
                    self.variables.append(node.args[0].value)
            self.generic_visit(node)

    extractor = VariableExtractor()
    extractor.visit(tree)
    return extractor.variables

def assemble1(form, u, v):
    # Extract variable names
    variables = extract_variables(form)
    print(f"Detected variables: {variables}")

    # Map variables to tensor outputs
    outputs = {var: u(var).requires_grad_(True) for var in variables}
    
    # Evaluate the function
    result = form(lambda var: outputs[var]).sum()

    # Compute gradients with respect to each variable
    grads = {}
    result.backward()
    for var, tensor in outputs.items():
        grads[var] = tensor.grad

    

    print(f"Result of the function: {result}")
    print(f"Gradients: {grads}")


