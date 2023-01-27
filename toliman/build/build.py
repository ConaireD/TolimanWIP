import os
import sys
import warnings
import tqdm
import requests

__author__ = "Jordan Dennis"

def build():
    print("Building `toliman`!")
    print("-------------------")

    if not _is_phoenix_installed():
        print("Installing phoenix...")
        _install_phoenix()
        print("Done!")

    print("Saving spectral model...")
    _simulate_alpha_cen_spectra()
    print("Done!")

    print("Saving background stars...")
    _simulate_background_stars()
    print("Done!")

    if not _is_mask_installed():
        print("Installing mask...")
        _install_mask()
        print("Done!")

    print("`toliman` built!")
