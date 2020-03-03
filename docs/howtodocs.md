# Welcome to mkdocs

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.

##  How to write docs with mkdocs

`MkDocs` are markdown documents, so the answer is easy: just use plain Markdown and, optionally, the supported extensions. More info [in the official docs](https://www.mkdocs.org/user-guide/writing-your-docs/#writing-with-markdown).

The theme we are using is `material`, which supports very fancy [extensions](https://squidfunk.github.io/mkdocs-material/extensions/admonition/).

For example, `admonitions` like this block:

!!! tip
    This is so cool huh? Check all styles [here](https://squidfunk.github.io/mkdocs-material/extensions/admonition/#types).

```
!!! tip
    This is so cool huh? Check all styles [here](https://squidfunk.github.io/mkdocs-material/extensions/admonition/#types).

```

Or citations:

> This is a very important finding.[^1]

> This is yet another finding.[^Rodríguez-Guerra and Pedregal, 1990]

[^1]: Lorem ipsum dolor sit amet, consectetur adipiscing elit.

[^Rodríguez-Guerra and Pedregal, 1990]: A kid named Jaime.

These are written with labels like this:

```
> This is a very important finding.[^1]

> This is yet another finding.[^Rodríguez-Guerra and Pedregal, 1990]

[^1]: Lorem ipsum dolor sit amet, consectetur adipiscing elit.

[^Rodríguez-Guerra and Pedregal, 1990]: A kid named Jaime.
```

### Docstrings

We are using [`mkdocstrings`](https://pawamoy.github.io/mkdocstrings/) for our docstrings, which deviate slightly from the more popular `numpydoc` syntax. Instead, it's closer to [Google-style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html). To sum up, this is a more or less complete example of the requested syntax:

    """
    A short description of this function.

    A longer description of this function.
    You can use more lines.

        This is code block,
        as usual.

        ```python
        s = "This is a Python code block :)"
        ```

    Arguments:
        param1: An integer?
        param2: A string? If you have a long description,
            you can split it on multiple lines.
            Just remember to indent those lines with at least two more spaces.
            They will all be concatenated in one line, so do not try to
            use complex markup here.

    Note:
        We omitted the type hints next to the parameters names.
        Usually you would write something like `param1 (int): ...`,
        but `mkdocstrings` gets the type information from the signature, so it's not needed here.

    Exceptions are written the same.

    Raises:
        OSError: Explain when this error is thrown.
        RuntimeError: Explain as well.
            Multi-line description, etc.

    Let's see the return value section now.

    Returns:
        A description of the value that is returned.
        Again multiple lines are allowed. They will also be concatenated to one line,
        so do not use complex markup here.

    Note:
        Other words are supported:

        - `Args`, `Arguments`, `Params` and `Parameters` for the parameters.
        - `Raise`, `Raises`, `Except`, and `Exceptions` for exceptions.
        - `Return` or `Returns` for return value.

        They are all case-insensitive, so you can write `RETURNS:` or `params:`.
    """

### More docstring examples

More examples, with and without types:

```python
def function(arg1, kwarg=None):
    """
    Example function

    Parameters:
        arg1 (dict): Some description for this argument.
            This type (in parenthesis) is ignored.
        kwarg: Some more descriptions

    Returns:
        A description for the returned value

    __Examples__

    The triple quotes below should be backticks (`)
    '''python
    2 + 2 == 4
    # True
    '''
    """
    pass


def function_with_types(
    arg1: dict, kwarg: "type hints can be whatever" = None
) -> tuple:
    """
    Example function. Types will be inferred from type hints!

    Parameters:
        arg1: Yeah
        kwarg: None

    Returns:
        Something
    """
    pass
```