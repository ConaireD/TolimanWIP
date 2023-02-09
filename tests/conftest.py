import pytest
import os
import shutil
import typing

class fixture(typing.Generic[typing.TypeVar("T")]): pass 

rm: callable = lambda root: shutil.rmtree(root) if os.path.isdir(root) else None 
mkdir: callable = lambda root: os.makedirs(root) if not os.path.isdir(root) else None

@pytest.fixture
def rmdir(root: str) -> None:
    """
    Ensures that tests are independent by deleting files before and after.

    Parameters
    ----------
    root: str
        The directories that contain the test files.
    """
    rm(root)
    yield
    rm(root)

@pytest.fixture
def make_fake_csv(
        root: str, 
        number_of_rows: int,
        number_of_columns: int,
        rmdir: fixture[None],
    ) -> None:
    """
    Create a phony csv file for testing.

    Fixtures
    --------
    rmdir: fixture[None]
        Ensure files are only extant during tests.

    Parameters
    ----------
    root: str
        The directory to populate with the csv.
    number_of_rows: int
        The number of the rows in the csv.
    number_of_columns: int
        The number of columns in the csv.
    """
    _: None = mkdir(root)
    file_name: str = "{}/fake.csv".format(root)
    with open(file_name, "w") as file:
        file.write(
            os.linesep.join(
                [
                    ",".join(
                        "header{}".format(column_header) 
                        for column_header in range(number_of_columns)
                    ),
                ] + [
                    ",".join("{}".format(column) 
                        for column in range(number_of_columns)
                    )
                    for row in range(number_of_rows)
                ]
            )
        )
    return file_name            
    
@pytest.fixture
import jax.lax as jl
import jax.numpy as np
import functools as ft
import jax
import matplotlib as mpl
import matplotlib.pyplot as plt

def get_cartesian_pixel_coordinates(pixels: int) -> float:
    translation: float = (pixels - 1) / 2.
    return jl.concatenate([
        jl.broadcasted_iota(float, (1, pixels, pixels), 1),
        jl.broadcasted_iota(float, (1, pixels, pixels), 2),       
    ], 0) - translation

def circular_aperture(cartesian_coordinates: float, radius: int) -> float:
    pythag: float = jl.integer_pow(cartesian_coordinates, 2)
    radii: float = jl.sqrt(jl.reduce(pythag, 0., jl.add, (0,)))
    return jl.lt(radii, radius).astype(float)

def make_fake_aperture(pixels: int) -> float:
    cartesian_coordinates: float = get_cartesian_pixel_coordinates(128)
    radius: float = pixels / 8.
    return circular_aperture(cartesian_coordinates, radius)

def make_fake_psf(pixels: int) -> float:
    aperture: float = make_fake_aperture(pixels)
    edge_zero_psf: float = jl.abs(jl.fft(aperture, "FFT", aperture.shape))
    return np.roll(edge_zero_psf, (pixels / 2, pixels / 2), axis=(0, 1))
