import numpy as np
import sympy as sy
import utils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from copulas import copula
from margins import margin


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

    def __init__(self, type='independent', para=[]):
        '''
        '''
        self.copula  = copula[type]()
        self.copula_para = para  

        self.margin = []
        self.margin_para = []
        self.N = 0

    def add_margin(self,type='uniform', para=[]):
        '''
        '''
        self.margin.append( margin[type]('_'+str(self.N)) )
        self.margin_para.append(para)
        self.N+=1

    def fit(self, X):
        '''
        '''
        # fit margin and compute U
        U = np.zeros_like(X)
        margin_para = []
        for i in range(X.shape[1]):

            F, para_F = self.margin[i]
            values = utils.fit_margin(F, para_F, X[:,i])
            margin_para.append(values)

            for p,val in zip(para_F[1:],values):
                F = F.subs(p,val)

            F_ = sy.lambdify(para_F[0],F,'numpy')
            U[:,i] = F_(X[:,i])
        self.margin_para = np.array(margin_para)

        # fit copula
        C, para_C = self.copula
        values = utils.fit_copula(C, para_C, U)
        self.copula_para=values



        return self

    def generate_x(self, N=1000):
        '''
        '''
        y = sy.symbols('y')
        C, para_C = self.copula
        u0 = para_C[0]
        u1 = para_C[1]
        d = para_C[2:]

        C_inv = sy.solve(sy.Eq(sy.diff(C,u0),y),u1)[0]
        C_inv = sy.lambdify((u0,y,d),C_inv,'numpy') 

        M_inv = []
        for M,v in zip(self.margin,self.margin_para):

            u = sy.symbols('u')
            F, para_F = M
            x0 = para_F[0]
            p = para_F[1:]
            
            F_inv = sy.solve(sy.Eq(F,u),x0)[0]
            F_inv = sy.lambdify((u,p),F_inv,'numpy') 
            M_inv.append(F_inv)

        X = []
        for i in range(N):
            u0, y = np.random.uniform(size=2)

            u1 = C_inv(u0,y,self.copula_para)

            x0 = M_inv[0](u0,self.margin_para[0])
            x1 = M_inv[1](u1,self.margin_para[1])
            X.append([x0,x1])

        return np.array(X)

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
