import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


def kde2D(xpts, ypts, nx=15, ny=None, yscale=1.0,
          xmin=None, xmax=None, ymin=None, ymax=None,
          ax=None, cmap='viridis', vmin=0, vmax=1,
          overplot_sample=0, overplot_color='white', 
          alpha=0.2, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    
    xmin = xmin if xmin is not None else np.min(xpts)
    xmax = xmax if xmax is not None else np.max(xpts)
    ymin = ymin if ymin is not None else np.min(ypts)
    ymax = ymax if ymax is not None else np.max(ypts)

    ny = ny if ny is not None else nx
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    kde = KernelDensity(**kwargs)
    kde.fit(np.vstack([xpts, yscale*ypts]).T)
    xy_sample = np.vstack([X.ravel(), yscale*Y.ravel()]).T
    Z = np.exp(kde.score_samples(xy_sample))
    dZ = np.max(Z)-np.min(Z)
    vmin=np.min(Z)+vmin*dZ
    vmax=np.max(Z)-(1-vmax)*dZ
    Z = Z.reshape(X.shape)

    ax.contourf(X, Y, Z, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)

    if overplot_sample == 'all' or overplot_sample > 0:
        mask = (xpts > xmin) & (xpts < xmax) & (ypts > ymin) & (ypts < ymax)
        xarr = np.array(xpts)[mask]
        yarr = np.array(ypts)[mask]
        if overplot_sample == 'all':
            sample = np.ones_like(xarr).astype(bool)
        else:
            sample = np.random.choice(xarr.shape[0],overplot_sample)
        ax.scatter(xarr[sample], yarr[sample], color=overplot_color, s=2, alpha=alpha)