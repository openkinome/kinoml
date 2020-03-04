"""
Example module to show how docstrings are written for
mkdocs + mkdocstrings
"""


def example_function(arg1, kwarg=None) -> object:
    """
    Example function to demonstrate how APIs are rendered

    Parameters:
        arg1 (dict): Some description for this argument.
            This type (in parenthesis) is ignored.
        kwarg: Some more descriptions

    Returns:
        A description for the returned value

    __Examples__

    This can be automatically tested with `pytest --doctest-modules`!
    Syntax might change subtly in the future.
    Check https://github.com/pawamoy/mkdocstrings/issues/52

    ```python
    >>> 2 + 2 == 4
    True  # this passes pytest
    >>> 2 + 2 == 5
    True  # this fails pytest

    ```
    """
    pass


def example_function_with_type_hints(arg1: dict, kwarg: "whatever" = None) -> object:
    """
    Example function to demonstrate how APIs are rendered

    Parameters:
        arg1: Some description for this argument.
        kwarg: Some more descriptions

    Returns:
        A description for the returned value

    __Examples__

    This can be automatically tested with `pytest --doctest-modules`!
    Syntax might change subtly in the future.
    Check https://github.com/pawamoy/mkdocstrings/issues/52

    ```python
    >>> 2 + 2 == 4
    True  # this passes pytest
    >>> 2 + 2 == 5
    True  # this fails pytest

    ```
    """
    pass
