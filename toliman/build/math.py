def normalise(arr: float) -> float:
    """
    Rescale and array onto [0, 1].

    Parameters
    ----------
    arr: float
        Any array.

    Returns
    -------
    arr: float
        An array of floating point numbers over the range [0, 1].
    """
    return (arr - arr.min()) / arr.ptp()
        
def angstrom_to_m(angstrom: float) -> float:
    """
    Convert an array that is in angstrom to meters.

    Parameters
    ----------
    angstrom: float, angstrom
        An array of measurements.

    Returns
    -------
    meters: float, meters
        An array of measurements.
    """
    m_per_angstrom: float = 1e-10
    return m_per_angstrom * angstrom
