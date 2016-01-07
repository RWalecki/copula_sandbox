import sympy as sy

def normal():
    '''
    '''
    v, v0, s = sy.symbols('v, u0, s')
    V = (v-v0) / (s * sy.sqrt(2))
    F = 0.5 * ( 1 + sy.erf(V) )
    return F, [v, v0, s]

def uniform():
    '''
    '''
    v, v0, s = sy.symbols('v, v0, s')
    F = (v-v0)/s
    return F, [v, v0, s]

def sigmoid():
    '''
    '''
    v, v0, s = sy.symbols('v, v0, s')
    V = (v-v0)/s
    F = 1/(1+sy.exp(-V))
    return F, [v, v0, s]
