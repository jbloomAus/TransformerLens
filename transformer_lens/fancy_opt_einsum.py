from fancy_einsum import convert_equation
from opt_einsum import contract


def fancy_opt_einsum(equation: str, *operands):
    """
    Variation on fancy opt einsum that uses opt_einsum for the contraction.
    
    Evaluates the Einstein summation convention on the operands.
    
    See: 
      https://pytorch.org/docs/stable/generated/torch.einsum.html
      https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
      https://optimized-einsum.readthedocs.io/en/stable/index.html
    """
    new_equation = convert_equation(equation)
    return contract(new_equation, *operands)