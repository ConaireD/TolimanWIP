import matplotlib.pyplot as plt
import matplotlib as mpl
import dLux as dl

mpl.rcParams["text.usetex"] = True
mpl.rcParams["image.cmap"] = "inferno"

coordinates = dl.utils.get_pixel_coordinates(256, 1. / 256.)

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
neg_rad_circ = dl.CircularAperture(0., 0., -1., occulting=True, softening=True)

fig = plt.figure()

plt.imshow(occ_soft_circ._aperture(coordinates))
ann_aper = dl.AnnularAperture()
rect_aper = dl.RectangularAperture()
sq_aper = dl.SquareAperture()
even_unif_spider = dl.EvenUniformSpider()
