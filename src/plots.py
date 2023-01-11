import matplotlib as mpl 
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True


def plot_im_on_fig(image: float, fig: object, cm: object = plt.cm.inferno) -> (object, object):
    axes: object = fig.add_axes([0., 0., .95, .95])
    _: list = axes.axis("off")
    cmap: object = axes.imshow(image, cmap=cm)
    return fig, cmap


def plot_im_with_cax_on_fig(image: float, fig: object, cm: object = plt.cm.inferno) -> object:
    fig, cmap = plot_im_on_fig(image, fig, cm)
    caxes: object = fig.add_axes([.95, 0., .05, .95])
    cbar: object = fig.colorbar(cmap, cax=caxes)
    _: None = cbar.outline.set_visible(False)
    return fig


def plot_ims_with_shared_cbar(ims: float, /, shape: tuple = None, cm: object = plt.cm.seismic) -> None:     
    fig: object = plt.figure(figsize=tuple(map(lambda x: x * 4., shape)))
    subfigs: object = fig.subfigures(*shape)
    rows: int = shape[0]
    cols: int = shape[1]
    
    for i in range(rows * cols):
        is_end: bool = ((i + 1) % cols == 0)
            
        if is_end:
            plot_im_with_cax_on_fig(ims[i], subfigs[i])
        else: 
            plot_im_on_fig(ims[i], subfigs[i])
        
    return fig
