import dLux as dl
import jax.numpy as np 

alpha_cen_a_ra = 219.902540961
alpha_cen_a_dec = -60.832666145
alpha_cen_a_gmag = 15.35

alpha_cen_b_ra = 219.902540961
alpha_cen_b_dec = -60.833410210
alpha_cen_b_gmag = 18.16

alpha_cen_ra = (alpha_cen_a_ra + alpha_cen_b_ra) / 2.
alpha_cen_dec = (alpha_cen_a_dec + alpha_cen_b_dec) / 2.
alpha_cen_contrast = np.exp(-0.4 * (alpha_cen_a_gmag - alpha_cen_b_gmag))

# So I can calculate the flux ratios of the stars using the formula 
# 
# m_1 - m_2 = - 2.5 log (f_1 / f_2) => f_1 / f_2 = exp(-0.4 * (m_1 - m_2))
# 
# I will work in units of the flux from Alpha Centauri A. Now this is 
# a triplet of stars so I am not sure if the functionality to model 
# this exists. If not I will build it. 
#
# Note: Alpha Centauri C is not going to be included in the image. 
# TODO: When I search Alpha Centuari I get different results to searching
#       each memeber separately. This is a behaviour that needs to be 
#       investigated. I want to use the midpoint of chord connecting 
#       Alpha Centuari A and B as the center of the two arcminute search. 
#
# So @peter has said that only stars within ten arcseconds of the science 
# target are likely to matter (23/11/2022). 

print(alpha_cen_ra)
print(alpha_cen_dec)

#alpha_centuari = dl.BinarySource(position=(0., 0.), )
