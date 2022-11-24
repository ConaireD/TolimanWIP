import dLux as dl

coordinates = dl.utils.get_pixel_coordinates(256, 1. / 256.)

circ_aper = dl.CircularAperture(0., 0., 1., occulting=True, softening=True)
circ_aper = dl.CircularAperture(0., 0., 1., occulting=True, softening=False)
circ_aper = dl.CircularAperture(0., 0., 1., occulting=True, softening=True)
circ_aper = dl.CircularAperture(0., 0., 1., occulting=True, softening=True)
ann_aper = dl.AnnularAperture()
rect_aper = dl.RectangularAperture()
sq_aper = dl.SquareAperture()
even_unif_spider = dl.EvenUniformSpider()
