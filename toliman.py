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
#
# So the spectral information is not yet ready to be integrated into 
# the model (23/11/2022). The ideal case in my mind is that we are 
# able to model using the Spectrum <=> Filter pair of interacting 
# parts. I would like to test the sensitivity of the model to the 
# wavelengths and weights that are passed into the system. 

tol_filt_min = 595e-09
tol_filt_max = 695e-09
tol_filt_wavels = np.linspace(tol_folt_min, tol_filt_max, 10) 

alpha_centuari = dl.BinarySource(
    position = np.array([0., 0.]), 
    flux = np.array(1.), 
    separation = np.array(alpha_cen_sep * np.pi / 180),
    position_angle = alpha_cen_pos_ang,
    contrast = np.array(alpha_cen_con),
    wavelengths = tol_filt_wavels)

# TODO: Add the background stars. I wonder if long term we are going to 
#       care about their spectra as well. I imagine that this is the 
#       case. This means that I will need to hunt down the relevant spectra. 
#       This is something that I will want to do in a way that is automated. 
#       We need to have all this information so we cam wprk out what level 
#       of detail can be pruned. This s confusing actually. Should I move 
#       this stuff into my notes file? Probably. 

# This is just examining the mask. There are two of them and one is 
# labelled as ".._sidelobes" so I want to check what the difference is. 

