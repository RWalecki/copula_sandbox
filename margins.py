import sympy as sy


def sigmoid(u0=0,s=1):
    '''
    '''
    u_, v_, v0_, s_ = sy.symbols('u, v, u0, s')

    # CDF:
    ###############################################
    V = (v_-v0_)/s_
    CDF = 1/(1+sy.exp(-V))
    ###############################################

    CDF_inv = sy.solve(sy.Eq(CDF,u_),v_)[0]
    CDF_inv = sy.lambdify((u_,v0_,s_),CDF_inv) 

    def out(u):return CDF_inv(u,u0,s)
    return out

def normal(u0=0,s=1):
    '''
    '''
    u_, v_, v0_, s_ = sy.symbols('u, v, u0, s')

    # CDF:
    ###############################################
    V = (v_-v0_) / (s_ * sy.sqrt(2))
    CDF = 0.5 * ( 1 + sy.erf(V) )
    ###############################################

    CDF_inv = sy.solve(sy.Eq(CDF,u_),v_)[0]
    CDF_inv = sy.lambdify((u_,v0_,s_),CDF_inv) 

    def out(u):return CDF_inv(u,u0,s)
    return out

def uniform(u0=0,s=1):
    '''
    '''
    u_, v_, v0_, s_ = sy.symbols('u, v, u0, s')

    # CDF:
    ###############################################
    CDF = (v_-v0_)/s_
    ###############################################

    CDF_inv = sy.solve(sy.Eq(CDF,u_),v_)[0]
    CDF_inv = sy.lambdify((u_,v0_,s_),CDF_inv) 

    def out(u):return CDF_inv(u,u0,s)
    return out
