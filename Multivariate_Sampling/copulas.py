import sympy as sy

def independent():
    '''
    '''
    u1, u2, d = sy.symbols('u1, u2, d')

    # Multinomial CDF
    ##################################################
    C = u1 * u2
    ##################################################

    return C, [u1, u2]

def frank():
    '''
    '''
    u1, u2, d = sy.symbols('u1, u2, d')

    U1 = sy.exp( -d * u1 ) - 1
    U2 = sy.exp( -d * u2 ) - 1
    D = sy.exp( -d ) - 1
    C = 1+U1*U2/D
    C = -1/(d) * sy.log(C)

    return C, [u1, u2, d]
