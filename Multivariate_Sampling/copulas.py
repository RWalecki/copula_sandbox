import sympy as sy

u1, u2, d   = sy.symbols('u1, u2, d')

def Independent():
    '''
    '''
    u1, u2, v2 = sy.symbols('u1, u2, v2')

    # Multinomial CDF
    ##################################################
    C = u1 * u2
    ##################################################

    CDF_inv = sy.solve(sy.Eq(sy.diff(C,u1),v2),u2)[0]
    CDF_inv = sy.lambdify((u1,v2),CDF_inv) 
    return CDF_inv 

def Frank(theta=4):
    '''
    '''
    u1, u2, v2, d = sy.symbols('u1, u2, v2, d')

    # Multinomial CDF
    ##################################################
    U1 = sy.exp( -d * u1 ) - 1
    U2 = sy.exp( -d * u2 ) - 1
    D = sy.exp( -d ) - 1
    C = 1+U1*U2/D
    C = -1/(d) * sy.log(C)
    ##################################################

    CDF_inv = sy.solve(sy.Eq(sy.diff(C,u1),v2),u2)[0]
    CDF_inv = sy.lambdify((u1,v2,d),CDF_inv) 
    def out(u1,v2):return CDF_inv(u1,v2,theta)
    return out
