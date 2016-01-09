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
