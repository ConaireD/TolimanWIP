import matplotlib.pyplot as plt
import matplotlib as mpl
import dLux as dl

mpl.rcParams["text.usetex"] = True
mpl.rcParams["image.cmap"] = "inferno"
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.left"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.bottom"] = False


coordinates = dl.utils.get_pixel_coordinates(24, 2. / 24.)

# My philosphy with the following trest cases is that each feature should
# only be tested once. In other words, I do not need to test every combination 
# of values just enough that any edge cases/branches are tested. Otherwise 
# there would be a vast number of outputs to check. As with all unit testing 
# the idea that only one thing is tested at a time is also used.
occ_soft_circ = dl.CircularAperture(0., 0., 1., occulting=True, softening=True)
occ_circ = dl.CircularAperture(0., 0., 1., occulting=True, softening=False)
soft_circ = dl.CircularAperture(0., 0., 1., occulting=False, softening=True)
circ = dl.CircularAperture(0., 0., 1., occulting=False, softening=False)
pos_y_circ = dl.CircularAperture(0., 1., 1., occulting=True, softening=True)
pos_x_circ = dl.CircularAperture(1., 0., 1., occulting=True, softening=True)
neg_y_circ = dl.CircularAperture(0., -1., 1., occulting=True, softening=True)
neg_x_circ = dl.CircularAperture(-1., 0., 1., occulting=True, softening=True)

fig, axes = plt.subplots(3, 3)
fig.suptitle("Circular Aperture Tests")
axes[0][0].set_title("Occulting, Soft Edged")
_map = axes[0][0].imshow(occ_soft_circ._aperture(coordinates))
axes[0][0].set_xticks([])
axes[0][0].set_yticks([])
fig.colorbar(_map)
axes[0][1].set_title("Occulting, Hard Edged")
_map = axes[0][1].imshow(occ_circ._aperture(coordinates))
axes[0][1].set_xticks([])
axes[0][1].set_yticks([])
fig.colorbar(_map)
axes[0][2].set_title("Not Occulting, Soft Edged")
_map = axes[0][2].imshow(soft_circ._aperture(coordinates))
axes[0][2].set_xticks([])
axes[0][2].set_yticks([])
fig.colorbar(_map)
axes[1][0].set_title("Not Occulting, Hard Edged")
_map = axes[1][0].imshow(circ._aperture(coordinates))
axes[1][0].set_xticks([])
axes[1][0].set_yticks([])
fig.colorbar(_map)
axes[1][1].set_title("Positive $y$ Translation (Occ., Soft.)")
_map = axes[1][1].imshow(pos_y_circ._aperture(coordinates))
axes[1][1].set_xticks([])
axes[1][1].set_yticks([])
fig.colorbar(_map)
axes[1][2].set_title("Positive $x$ Translation (Occ., Soft.)")
_map = axes[1][2].imshow(pos_x_circ._aperture(coordinates))
axes[1][2].set_xticks([])
axes[1][2].set_yticks([])
fig.colorbar(_map)
axes[2][0].set_title("Negative $y$ Translation (Occ., Soft.)")
_map = axes[2][0].imshow(neg_y_circ._aperture(coordinates))
axes[2][0].set_xticks([])
axes[2][0].set_yticks([])
fig.colorbar(_map)
axes[2][1].set_title("Negative $x$ Translation (Occ., Soft.)")
_map = axes[2][1].imshow(neg_x_circ._aperture(coordinates))
axes[2][1].set_xticks([])
axes[2][1].set_yticks([])
fig.colorbar(_map)
axes[2][2].set_xticks([])
axes[2][2].set_yticks([])
plt.show()

ann_aper = dl.AnnularAperture()
rect_aper = dl.RectangularAperture()
sq_aper = dl.SquareAperture()
even_unif_spider = dl.EvenUniformSpider()
