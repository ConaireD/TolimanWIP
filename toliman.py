import dLux as dl
import jax.numpy as np 

alpha_cen_a_ra = 219.902540961 # degrees
alpha_cen_a_dec = -60.833410210 # degrees
alpha_cen_a_gmag = 18.16 # mag
alpha_cen_a_pos = np.array([alpha_cen_a_ra, alpha_cen_a_dec]) # degrees

alpha_cen_b_ra = 219.902540961 # degrees
alpha_cen_b_dec = -60.832666145 # degrees
alpha_cen_b_gmag = 15.35 # mag
alpha_cen_b_pos = np.array([alpha_cen_b_ra, alpha_cen_b_dec]) # degrees

alpha_cen_ra = (alpha_cen_a_ra + alpha_cen_b_ra) / 2.
alpha_cen_dec = (alpha_cen_a_dec + alpha_cen_b_dec) / 2.
alpha_cen_con = np.exp(-0.4 * (alpha_cen_a_gmag - alpha_cen_b_gmag))
alpha_cen_sep_vec = alpha_cen_a_pos - alpha_cen_b_pos 
alpha_cen_sep = np.abs(alpha_cen_sep_vec)
alpha_cen_pos_ang = np.arctan(alpha_cen_sep_vec[0] / alpha_cen_sep_vec[1])

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

alpha_cen_spectra = 

alpha_centuari = dl.BinarySource(
    position = np.array([0., 0.]), 
    flux = np.array(1.), 
    separation = np.array(alpha_cen_sep * np.pi / 180),
    position_angle = alpha_cen_pos_ang,
    contrast = np.array(alpha_cen_con),
    spectrum = )
