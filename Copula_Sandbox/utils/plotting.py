import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sympy as sy

def plot_summary(model, samples, path):
    '''
    '''

    ##################################################################
    # get plot range
    ##################################################################
    if samples==None:
        test_samples = model.generate_x(1000)
    else:
        test_samples = samples

    min_, max_ = np.min(test_samples,0), np.max(test_samples,0)
    extent = [min_[0],max_[0],min_[1],max_[1]]
    ##################################################################


    ##################################################################
    # compute P(x1,x2|...) and c(u1,u2|...)
    ##################################################################
    P = model.C
    for u in model.U:P = sy.diff(P,u)
    c = P

    # substitute U with margins
    for m in range(model.dim):
        P = P.subs(model.U[m], model.F[m])

    # multiply P with marginal probability (from chain rule)
    for m in range(model.dim):
        P = P * sy.diff(model.F[m],model.X[m])

    # substitute model parameter (copula)
    P = P.subs(model.D,model.C_para)
    c = c.subs(model.D,model.C_para)

    # substitute model parameter (margins)
    for m in range(model.dim): # loop over margins
        for p in range(len(model.P[m])): # loop over margin parameter
            P = P.subs(model.P[m][p],model.F_para[m][p])

    jpdf_c = sy.lambdify(model.U,c,'numpy')
    jpdf = sy.lambdify(model.X,P,'numpy')
    ##################################################################


    ##################################################################
    # compute P(xi|...) to each margin
    ##################################################################
    pdf = []
    for m in range(model.dim):
        _F = model.F[m]
        for p in range(len(model.P[m])): # loop over margin parameter
            _F = _F.subs(model.P[m][p],model.F_para[m][p])
        _pdf = sy.lambdify(model.X[m],sy.diff(_F,model.X[m]),'numpy') 
        pdf.append(_pdf)
    ##################################################################


    ##################################################################
    # compute CDF(xi|...) to each margin
    ##################################################################
    cdf = []
    for m in range(model.dim):
        _F = model.F[m]
        for p in range(len(model.P[m])): # loop over margin parameter
            _F = _F.subs(model.P[m][p],model.F_para[m][p])
        _cdf = sy.lambdify(model.X[m],_F,'numpy') 
        cdf.append(_pdf)
    ##################################################################







    fig = plt.figure()
    gs1 = gridspec.GridSpec(2, 2)

    ##################################################################
    # 1) bivariate density function
    ##################################################################
    ax = plt.subplot(gs1[:-1, 1:])
    if samples!=None:
        plt.scatter(test_samples.T[0],test_samples.T[1],c='k',alpha=0.5)
    ax = plot_fun2(ax, jpdf, extent)
    ax.set_title('P(x_0,x_1)')
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ##################################################################

    ##################################################################
    # 2) density function of margin 0
    ##################################################################
    ax1 = plt.subplot(gs1[-1, 1:])
    if samples!=None:
        ax1.hist(test_samples.T[0], bins=10, normed=True)
    ax1.set_title('p(x_0)')
    ax1 = plot_fun1(ax1,pdf[0],extent[:2])
    ax1.set_xlabel('x_0')
    ax1.set_yticks([]) 
    ##################################################################

    ##################################################################
    # 3) density function of margin 1
    ##################################################################
    ax0 = plt.subplot(gs1[:-1, 0])
    if samples!=None:
        ax0.hist(test_samples.T[1], bins=10, orientation="horizontal",normed=True)
    ax0.set_title('p(x_1)')
    ax0 = plot_fun1(ax0,pdf[1],extent[2:],rot=1)
    ax0.set_ylabel('x_1')
    ax0.set_xticks([]) 
    ##################################################################

    ##################################################################
    # 4) copula density function
    ##################################################################
    ax3 = plt.subplot(gs1[-1, 0])
    if samples!=None:
        u = model.transform_u(samples)
        plt.scatter(u[:,0],u[:,1],c='k',alpha=0.5)
    ax3 = plot_fun2(ax3,jpdf_c,[0,1,0,1])
    ax3.set_title('c(u_0,u_1)')
    ax3.set_xticks([]) 
    ax3.set_yticks([]) 
    ##################################################################


    plt.savefig(path, transparent=True, bbox_inches='tight')

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


    plt.imshow(zz,origin='lower',extent=extent)

    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
    return ax
