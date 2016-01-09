import matplotlib.pyplot as plt
from scipy import ndimage


def plot_fun1(ax, fun, extent, rot=False, aspect=1, pix=100):
    '''
    '''
    xx = np.linspace(extent[0], extent[1], pix)
    yy = []
    for x in xx:yy.append(fun(x))
    if rot:
        plt.plot(yy,xx,linewidth=5.0)
    else:
        plt.plot(xx,yy,linewidth=5.0)
    extent = ax.get_xlim()+ax.get_ylim()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/1)

    return ax

def plot_fun2(ax, fun, extent, aspect=1, pix=100):
    '''
    '''
    xx, yy = np.meshgrid(
            np.linspace(extent[0], extent[1], pix),
            np.linspace(extent[2], extent[3], pix)
            )

    zz = []
    for x,y in zip(xx.ravel(),yy.ravel()):
        zz.append(fun(x,y))
    zz = np.array(zz)
    zz = zz.reshape(xx.shape)
    print zz.min()
    print zz.max()
    print zz.std()


    plt.imshow(zz,origin='lower',extent=extent)

    values = np.sort(zz.ravel())
    c0 = len(values)*0.01
    c1 = len(values)*0.99

    plt.clim(values[c0],values[c1])

    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
    return ax

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
