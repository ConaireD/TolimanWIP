import abc

__author__ = "Jordan Dennis"
__all__ = ["contains_instance", "CollectionInterface"]

def contains_instance(collection: list, instance_of: type) -> bool:
    """
    Check to see if a list constains an element of a certain type.

    Parameters
    ----------
    collection: list
        The list to search.
    instance_of: type
        The type to check for.

    Returns
    -------
    contains: bool
        True if _type was found else False.
    """
    if collection:
        for element in collection:
            if isinstance(element, instance_of):
                return True
    return False


class CollectionInterface(abc.ABC):
    @abc.abstractmethod
    def to_optics_list(self: object) -> list:
        """
        Get the optical elements that make up the object as a list.

        Returns
        -------
        optics: list
            The optical layers in order in a list.
        """

    @abc.abstractmethod
    def insert(self: object, optic: object, index: int) -> object:
        """
        Add an additional layer to the optical system.

        Parameters
        ----------
        optic: object
            A `dLux.OpticalLayer` to include in the model.
        index: int
            Where in the list of layers to add optic.

        Returns
        -------
        toliman: TolimanOptics
            A new `TolimanOptics` instance with the applied update.
        """

    @abc.abstractmethod
    def remove(self: object, index: int) -> object:
        """
        Take a layer from the optical system.

        Parameters
        ----------
        index: int
            Where in the list of layers to remove an optic.

        Returns
        -------
        toliman: TolimanOptics
            A new `TolimanOptics` instance with the applied update.
        """

    @abc.abstractmethod
    def append(self: object, optic: object) -> object:
        """
        Place a new optic at the end of the optical system.

        Parameters
        ----------
        optic: object
            The optic to include. It must be a subclass of the
            `dLux.OpticalLayer`.

        Returns
        -------
        optics: object
            The new optical system.
        """

    @abc.abstractmethod
    def pop(self: object) -> object:
        """
        Remove the last element in the optical system.

        Please note that this differs from the `.pop` method of the
        `list` class because it does not return the popped element.

        Returns
        -------
        optics: object
            The optical system with the layer removed.
        """
