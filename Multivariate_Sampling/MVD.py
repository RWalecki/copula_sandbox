import numpy as np
import sympy as sy
import utils
import copulas
import margins


def adjustFigAspect(fig,aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .4*minsize/xsize
    ylim = .4*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)


class MVD():
    '''
    C copula function
    D copula parameter: d
    U copula variables: [u0,u1,u2...]

    M list of margin functions
    P list of margin parameter 
    X list of margin variables
    '''
    def __init__(self, type='independent', dim = 2, para=None,verbose=0):
        '''
        '''
        if verbose:print 'initial Copula: '+type
        self.C, self.U, self.D = copulas.load(type,dim)
        if verbose:print self.C
        if verbose:print self.U
        if verbose:print self.D,para, '\n'
        self.C_type = type
        self.C_para = para

        self.F = [None] * dim
        self.P = [None] * dim
        self.X = [None] * dim
        self.F_type = [None] * dim
        self.F_para = [None] * dim
        self.dim = dim
        self.verbose = verbose

    def add_margin(self,dim, type='uniform', para=[]):
        '''
        '''
        if self.verbose:print 'initial margin:',dim,type
        F, X, P = margins.load(type,dim)
        if self.verbose:print F
        if self.verbose:print X
        if self.verbose:print P,para, '\n'
    
        self.F[dim] = F
        self.P[dim] = P
        self.X[dim] = X
        self.F_para[dim] = para
        self.F_type[dim] = type

    def transform_u(self,samples):
        '''
        # compute multivariate data U belonging to unit hypercubek
        '''

        U = np.zeros_like(samples)
        for m in range(self.dim):
            F = sy.lambdify([self.X[m]]+self.P[m],self.F[m],'numpy')
            U[:,m] = F(samples[:,m],*self.F_para[m])

        return U

    def fit(self, samples):
        '''
        '''
        # fit margins
        for m in range(self.dim):
            res = utils.fitting.fit_margin(
                    self.F[m],
                    self.X[m],
                    self.P[m],
                    samples[:,m]
                    )
            self.F_para[m]=res
            if self.verbose==1:print res

        U = self.transform_u(samples)

        # fit copula 
        if self.C_type=='independent':return self
        res = utils.fitting.fit_copula(
                self.C, 
                self.U,
                self.D,
                U
                )
        if self.verbose==1:print res
        self.C_para=res



        return self

    def generate_x(self, N=1000):
        '''
        this sampling method works wit all margins,
        but only frank and independent copulas can be used
        and it is limited to 2 dimensions
        '''
        # compute marginal prob of u1
        P_U1 = sy.simplify(sy.diff(self.C,self.U[0]))

        # invert marginal prob of u1
        y = sy.symbols('y')
        tmp = sy.solve(sy.Eq(P_U1,y),self.U[1])
        inv_P_U1 = sy.lambdify((self.U[0],y,self.D),tmp,'numpy') 

        # invert margins 
        inv_F = {}
        for m in [0,1]:
            u = sy.symbols('u')
            tmp = sy.solve(sy.Eq(self.F[m],u),self.X[m])[0]
            inv_F[m] = sy.lambdify((u,self.P[m]),tmp,'numpy') 


        X = np.zeros((N,2))
        for i in range(N):
            u0, y = np.random.uniform(size=2)
            u1 = inv_P_U1(u0,y,self.C_para)
            X[i,0] = inv_F[0](u0,self.F_para[0])
            X[i,1] = inv_F[1](u1,self.F_para[1])

        return X

    def plot_model(self, samples=None):
        '''
        '''
        # print model elements 
        print self.C_type
        print self.C_para, '\n'
        for m in range(self.dim):
            print self.F_type[m]
            print self.F_para[m]
            print self.F_para[m], '\n'

        if self.dim!=2:
            print 'WARNING! only bivariate Copula models can be plotted'
            print 'FIX: set "dimensions=2" and try again'
            return self

        # compute copula probability density function 
        P = self.C
        for u in self.U:P = sy.diff(P,u)

        # substitute U with margins
        for m in range(self.dim):
            P = P.subs(self.U[m], self.F[m])

        # multiply P with marginal probability (from chain rule)
        for m in range(self.dim):
            P = P * sy.diff(self.F[m],self.X[m])

        # substitute model parameter
        P = P.subs(self.D,self.C_para)
        for m in range(self.dim): # loop over margins
            for p in range(len(self.P[m])): # loop over margin parameter
                P = P.subs(self.P[m][p],self.F_para[m][p])
        P = sy.lambdify(self.X,P,'numpy')

        # this is the jpdf
        x=np.linspace(-10,10,100)

        print P(x,x)
        # todo: marginal pdf 
        # todo: copula cdf 
        # todo: marginal cdf 
        # add scatter
        # deriva all equation ad add to plots
        return self



