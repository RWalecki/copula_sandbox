import numpy as np
import sympy as sy
import utils
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
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
    def __init__(self, type='independent', dim = 2, para=[],verbose=0):
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

    def visual_model(self,path,samples=None):
        '''
        plot Bivariate density
        '''
        C, para_C = self.copula
        u0 = para_C[0]
        u1 = para_C[1]
        d = para_C[2:]

        M0, para_M0 = self.margin[0]
        x0 = para_M0[0]
        p0 = para_M0[1:]
        m0 = sy.diff(M0,x0)

        M1, para_M1 = self.margin[1]
        x1 = para_M1[0]
        p1 = para_M1[1:]
        m1 = sy.diff(M1,x1)

        p = sy.diff(sy.diff(C,u0),u1)
        c = p.subs(u0,M0)
        c = c.subs(u1,M1)

        _jpdf = c * m0 * m1
        _p    = sy.lambdify((u0,u1,d),p,'numpy') 
        _jpdf = sy.lambdify((x0,x1,d,p0,p1),_jpdf,'numpy') 
        _pdf0 = sy.lambdify((x0,p0),m0,'numpy') 
        _pdf1 = sy.lambdify((x1,p1),m1,'numpy') 
        _cdf0 = sy.lambdify((x0,p0),M0,'numpy') 
        _cdf1 = sy.lambdify((x1,p1),M1,'numpy') 

        def p(u0,u1):return _p(u0,u1,self.copula_para)
        def pdf0(x):return _pdf0(x,self.margin_para[0])
        def pdf1(x):return _pdf1(x,self.margin_para[1])
        def cdf0(x):return _cdf0(x,self.margin_para[0])
        def cdf1(x):return _cdf1(x,self.margin_para[1])
        def jpdf(x0,x1):return _jpdf(x0,x1,
                self.copula_para,
                self.margin_para[0],
                self.margin_para[1]
                )

        if samples==None:test_samples = self.generate_x(1000)
        else:test_samples = samples

        min_, max_ = np.min(test_samples,0), np.max(test_samples,0)
        extent = [min_[0],max_[0],min_[1],max_[1]]

        fig = plt.figure()
        gs1 = gridspec.GridSpec(2, 2)

        ##################################################################
        # 1) bivariate density function
        ##################################################################
        ax = plt.subplot(gs1[:-1, 1:])
        if samples!=None:
            plt.scatter(test_samples.T[0],test_samples.T[1],c='k',alpha=0.5)
        ax = utils.plot_fun2(ax, jpdf,extent)
        ax.set_title('P(x_0,x_1)')
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        ##################################################################


        ##################################################################
        # 2) density function of margin 0
        ##################################################################
        ax0 = plt.subplot(gs1[:-1, 0])
        if samples!=None:
            ax0.hist(test_samples.T[1], bins=10, orientation="horizontal",normed=True)
        ax0.set_title('p(x_1)')
        ax0 = utils.plot_fun1(ax0,pdf1,extent[2:],rot=1)
        ax0.set_ylabel('x_1')
        ax0.set_xticks([]) 
        ##################################################################

        ##################################################################
        # 3) density function of margin 1
        ##################################################################
        ax1 = plt.subplot(gs1[-1, 1:])
        if samples!=None:
            ax1.hist(test_samples.T[0], bins=10, normed=True)
        ax1.set_title('p(x_0)')
        ax1 = utils.plot_fun1(ax1,pdf0,extent[:2])
        ax1.set_xlabel('x_0')
        ax1.set_yticks([]) 
        ##################################################################


        ##################################################################
        # 4) copula density function
        ##################################################################
        ax3 = plt.subplot(gs1[-1, 0])
        if samples!=None:
            u0,u1 = [],[]
            for x,y in test_samples:
                u0.append(cdf0(x))
                u1.append(cdf1(y))
            plt.scatter(u0,u1,c='k',alpha=0.5)
        ax3 = utils.plot_fun2(ax3,p,[0,1,0,1])
        ax3.set_title('c(u_0,u_1)')
        ax3.set_xticks([]) 
        ax3.set_yticks([]) 
        ##################################################################


        ##################################################################
        # save plots 
        ##################################################################
        adjustFigAspect(fig,aspect=1)
        plt.savefig('/tmp/tmp.pdf', transparent=True, bbox_inches='tight')
