import os
import sys
import warnings
import tqdm
import requests

__author__ = "Jordan Dennis"
__all__ = ["_is_phoenix_installed", "_install_phoenix", "_accumulate_path"]


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
