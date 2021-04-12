# How to write docs with Sphinx, MyST and Material Theme

We are using Sphinx for our documentation. However, instead of using the default RST,
you can also use Markdown syntax thanks to the [MyST parser](https://myst-parser.readthedocs.io/).
The theme is [Material for Sphinx](https://github.com/bashtage/sphinx-material/).

## Basics

- `cd docs && make livebuild` - Start the live-reloading docs server locally

Project layout:

    docs/
        index.md  # The documentation homepage.
        conf.py   # The configuration file
        ...       # Other markdown pages, images and other files.

We prefer using Markdown for the documentation, but the Python docstrings
use RST with [NumpyDoc](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard)
conventions. Check the existing docstrings for syntax examples.

## Supported extensions

MyST adds some [extra stuff](https://myst-parser.readthedocs.io/en/latest/using/syntax.html#)
on top of plain Markdown. Some examples:

### Admonitions

```{note}
This is so cool huh? Check all styles [here](https://docutils.sourceforge.io/docs/ref/rst/directives.html#specific-admonitions).
```

````md
```{note}
This is so cool huh? Check all styles [here](https://docutils.sourceforge.io/docs/ref/rst/directives.html#specific-admonitions).
```
````

### Footnotes

> This is a very important finding.[^1]

> This is yet another finding.[^jaimergp1990]

[^1]: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
[^jaimergp1990]: A kid named Jaime.

These are written with labels like this:

```md
> This is a very important finding.[^1]

> This is yet another finding.[^jaimergp1990]

[^1]: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
[^jaimergp1990]: A kid named Jaime.
```

### LaTeX

Either in blocks

$$
\frac{n!}{k!(n-k)!} = \binom{n}{k} * KinoML
$$

```latex
$$
\frac{n!}{k!(n-k)!} = \binom{n}{k} * KinoML
$$
```

or inline:

This my best equation ever: $p(x|y) = \frac{p(y|x)p(x)}{p(y)}$

```latex
This my best equation ever: $p(x|y) = \frac{p(y|x)p(x)}{p(y)}$
```

### Tabbed fences

:::{tabbed} Step 1

This is the step 1
:::

:::{tabbed} Step 2

```python
# This is the step 2 with python code highlighting
he = Element("Helium")
```

:::

:::{tabbed} Step 3

This is the step 3
:::

This line interrupts the fences and creates a new block of tabs

:::{tabbed} Step 4

```python
# This is the step 4 with python code highlighting

be = Element("Beryllium")
```

:::

Obtained with:

````
:::{tabbed} Step 1

This is the step 1
:::

::::{tabbed} Step 2
```python
# This is the step 2 with python code highlighting
he = Element("Helium")
```
::::

:::{tabbed} Step 3

This is the step 3
:::

This line interrupts the fences and creates a new block of tabs

:::{tabbed} Step 4
```python
# This is the step 4 with python code highlighting

be = Element("Beryllium")
```
:::

````

### Extra inline markup

| Code      | Result  |
| --------- | ------- |
| `==hey==` | ==hey== |
| `~~hey~~` | ~~hey~~ |
| `^^hey^^` | ^^hey^^ |
| `a^migo^` | a^migo^ |
| `-->`     | -->     |
