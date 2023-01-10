def plot_basis(basis: float, /, shape: tuple = None) -> None:
    """
    Plot a set of polynomial vectors that comprise a
    basis. 
    
    Parameters:
    -----------
    basis: tensor[float]
        The basis vectors arranged as (nvecs, npix, npix).
    rows: tuple[int]
        The number of rows and cols in the plotting grid.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    mpl.rcParams["image.cmap"] = "seismic"
    mpl.rcParams["text.usetex"] = True
    
    number_of_vecs: int = basis.shape[0]
    inches_per_col: int = 3
    inches_per_row: int = 3
    
    # TODO: Implement for odd cases using subfigures to pull 
    #       out the first basis vec into its own centred 
    #       of the same size as the others. 
    if not shape:
        if number_of_vecs % 2 == 0: # is even
            rows: int = 2
            cols: int = number_of_vecs // 2
        else:
            rows: int = 1
            cols: int = number_of_vecs
    else:
        rows: int = shape[0]
        cols: int = shape[1]
    
    if number_of_vecs > rows * cols:
        raise ValueError("Not enough subplots were provided.")
        
    width: int = cols * inches_per_col
    height: int = rows * inches_per_row
    
    min_of_basis: float = basis.min()
    max_of_basis: float = basis.max()
    
    fig = plt.figure(figsize = (width, height))
    axes = fig.subplots(rows, cols)
    
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.axis("off")
    
    for vec in range(number_of_vecs):
        vec_cmap: object = axes.flat[vec].imshow(
            basis[vec][:, ::-1], # Image coordinates
            vmin = min_of_basis, 
            vmax = max_of_basis
        )
        
    fig.subplots_adjust(
        left = 0, 
        right = 1.,
        bottom = .15
    )
    
    col_bar_axis: object = fig.add_axes([0., 0., 1., .08])
        
    col_bar: object = fig.colorbar(
        vec_cmap, 
        cax = col_bar_axis, 
        orientation = "horizontal"
    )
    
    col_bar.ax.tick_params(labelsize = 20)
    col_bar.outline.set_visible(False)
    
    return None


def plot_pupil(pupil: float) -> None:
    """
    Plots an aperture pupil in a nice way. 
    
    Parameters:
    -----------
    pupil: float
        The pupil as an array.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["text.usetex"]: bool = True
    mpl.rcParams["image.cmap"]: str = "Greys"

    width_in_inches: int = 4

    fig: object = plt.figure(figsize = (width_in_inches, width_in_inches))

    image_axes: object = fig.add_axes([0., 0., .9, .9])
    image_cmap: object = image_axes.imshow(pupil[:, ::-1], vmin = 0., vmax = 1.)

    image_axes.set_xticks([])
    image_axes.set_yticks([])
    image_axes.axis("off")

    cbar_axes: object = fig.add_axes([.95, 0., .05, .9])
    cbar: object = fig.colorbar(image_cmap, cax=cbar_axes)

    cbar.ax.tick_params(labelsize = 20)
    cbar.outline.set_visible(False)
    
    return None


def plot_pupil_with_aberrations(pupil: float, aberrations: float) -> None:
    """
    Plot a pupil that has aberrations.
    
    Parameters:
    -----------
    pupil: float
        The (nearly)-binary array representing the pupil.
    aberrations: float
        The optical path differences. 
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    mpl.rcParams["text.usetex"]: bool = True
    mpl.rcParams["image.cmap"]: str = "seismic"
    
    max_displacement: float = np.max(np.abs(aberrations))
    aberrated_pupil: float = pupil * aberrations.sum(axis = 0)
    width_in_inches: int = 4

    fig: object = plt.figure(figsize = (width_in_inches, width_in_inches))

    image_axes: object = fig.add_axes([0., 0., .9, .9])
    image_cmap: object = image_axes.imshow(
        aberrated_pupil[:, ::-1],
        vmin = - max_displacement,
        vmax = max_displacement
    )

    image_axes.set_xticks([])
    image_axes.set_yticks([])
    image_axes.axis("off")

    cbar_axes: object = fig.add_axes([.95, 0., .05, .9])
    cbar: object = fig.colorbar(image_cmap, cax=cbar_axes)

    cbar.ax.tick_params(labelsize = 20)
    cbar.outline.set_visible(False)
    
    return None 


def plot_psf(psf: float) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
     
    mpl.rcParams["text.usetex"]: bool = True
    
    fig: object = plt.figure(figsize=(4, 4))
    axes: object = fig.add_axes([0., 0., .95, .95])
    _: list = axes.axis("off")
    caxes: object = fig.add_axes([.95, 0., .05, .95])
    cmap: object = axes.imshow(psf, cmap=plt.cm.inferno)
    cbar: object = fig.colorbar(cmap, cax=caxes)
    _: None = cbar.outline.set_visible(False)
