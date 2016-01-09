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
    '''
    import matplotlib.pyplot as plt
    from utils import plot_fun2
    import numpy as np

    C, U, d = load('frank',2)
    D = 3
    print U
    print d
    y = sy.symbols('y')
    P = sy.simplify(sy.diff(C,U[0]))
    print sy.invert(P,y)

    _cdf = sy.lambdify([U,d],C)
    def cdf(u0,u1):return _cdf([u0,u1],D)
    print cdf(0.1,0.3)

    _pdf = sy.diff(sy.diff(C,U[0]),U[1])
    _pdf = sy.lambdify([U,d],_pdf)
    def pdf(u0,u1):return _pdf([u0,u1],D)

    f, ax = plt.subplots()
    m = 0.01
    plot_fun2(ax, pdf, [m,1-m,m,1-m], aspect=1)
    plt.savefig('/tmp/tmp.pdf', transparent=True, bbox_inches='tight')

    G, [t,d] = frank_G()
    from sympy.integrals import laplace_transform as lt
    y = sy.symbols('y')
    u = np.random.uniform(size=2)
    print u
    y = sy.symbols('y')
    G_inv = sy.solve(sy.Eq(G,y),t)[0].subs(y,t)

    f = sy.lambdify(t,G.subs(d,10),'numpy')
    x = np.linspace(0,1,100)
    print f(x)

    '''
