import sympy as sy

def independent():
    t, d = sy.symbols('t, d')
    G = -sy.log(t)
    return G, [t, d]

def frank():
    t, d = sy.symbols('t, d')
    A = sy.exp(-d*t)-1
    B = sy.exp(-d)-1
    G = -sy.ln(A/B)
    return G, [t, d]

def gumble():
    t, d = sy.symbols('t, d')
    G = (-sy.log(t))**d
    return G, [t, d]

def joe():
    t, d = sy.symbols('t, d')
    G = -sy.log(1-(1-t)**d)
    return G, [t, d]

def clyton():
    t, d = sy.symbols('t, d')
    G = (1/d) * (t**(-d)-1)
    return G, [t, d]

def ali_mikhail_haq():
    t, d = sy.symbols('t, d')
    A = 1-d*(1-t)
    G = sy.log(A/t)
    return G, [t, d]

def load( type, dim=2 ):
    '''
    '''
    gen = {
            'frank':frank,
            'independent':independent,
            'gumble':gumble,
            'joe':joe,
            'clyton':clyton,
            'ali_mikhail_haq':ali_mikhail_haq,
            }


    # load copula generator
    G, [t, d] = gen[type]()

    # initial copula variables u0, u1, ... un
    U = []
    for i in range(0, dim):
        U.append( sy.symbols( 'u'+str(i) ) )

    # initial generator argument: G(u0)+G(u1)+...+G(un)
    gen_arg = 0
    for u in U:gen_arg+=G.subs(t,u)

    # compute inverse generator
    y = sy.symbols('y')
    inv_G = sy.solve(sy.Eq(G,y),t)[0].subs(y,t)

    # compute copula: C_t = inv_G[ G(u0)+G(u1)+...+G(un) ]
    C = inv_G.subs(t,gen_arg)

    return C, U, d




if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np

    C, U, d = load('frank',2)
    D = 3
    print(U)
    print(d)
    y = sy.symbols('y')
    P = sy.simplify(sy.diff(C,U[0]))
