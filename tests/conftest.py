import pytest
import os
import shutil
import typing
import jax.lax as jl
import jax.numpy as np

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
def coordinates(pixels: int) -> float:
    """
    Generate pixel coordinates. 

    The implementation was optimised fairly severly for an intel machine 
    CPU instruction set. This was to ensure that the F in F.I.R.S.T was 
    met.This fixture is not compiled because the is not known at runtime.

    Parameters
    ----------
    pixels: int
        The psf will be (pixels, pixels) in size.
    """
    translation: float = (pixels - 1) / 2.
    x: float = jl.broadcasted_iota(float, (1, pixels, pixels), 1)
    y: float = jl.broadcasted_iota(float, (1, pixels, pixels), 2)
    return jl.concatenate([x, y], 0) - translation

@pytest.fixture
def make_airy_psf(pixels: int, coordinates: fixture[float]) -> float:
    """
    Generate an airy pattern.

    The implementation was optimised fairly severly for an intel machine 
    CPU instruction set. This was to ensure that the F in F.I.R.S.T was 
    met.This fixture is not compiled because the is not known at runtime.
   
    Fixtures
    --------
    coordinates: fixture[float]
        Create a set of para-axial image coordinates.

    Parameters
    ----------
    pixels: int
        The psf will be (pixels, pixels) in size. Indirectly parametrizes 
        coordinates.
    """
    pythag: float = jl.integer_pow(coordinates, 2)
    radii: float = jl.sqrt(jl.reduce(pythag, 0., jl.add, (0,)))
    radius: float = pixels / 8.
    aperture: float =  jl.lt(radii, radius).astype(float)
    edge_zero_psf: float = jl.abs(jl.fft(aperture, "FFT", aperture.shape))
    return np.roll(edge_zero_psf, (pixels / 2, pixels / 2), axis=(0, 1))
