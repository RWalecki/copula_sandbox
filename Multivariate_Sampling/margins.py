import sympy as sy

def normal(n='_0'):
    '''
    '''
    v, v0, s = sy.symbols('v'+n+',u0'+n+',s'+n)
    V = (v-v0) / (s * sy.sqrt(2))
    F = 0.5 * ( 1 + sy.erf(V) )
    return F, [v, v0, s]

def uniform(n='_0'):
    '''
    '''
    v, v0, s = sy.symbols('v'+n+',u0'+n+',s'+n)
    F = (v-v0)/s
    return F, [v, v0, s]

def sigmoid(n='_0'):
    '''
    '''
    v, v0, s = sy.symbols('v'+n+',u0'+n+',s'+n)
    V = (v-v0)/s
    F = 1/(1+sy.exp(-V))
    return F, [v, v0, s]

margin = {
        'normal':normal,
        'uniform':uniform,
        'sigmoid':sigmoid,
        }
