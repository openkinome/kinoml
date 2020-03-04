import sys
from pathlib import Path
from collections import defaultdict

import yaml

"""
Create stubs for all API reference *.md files
and propose a menu tree (you probably need to edit it to your liking)

Redefine docs and package below and execute from repository root
"""

docs = "apidocs"
package = "kinoml"
here = Path(package)
tree = defaultdict(list)

Path(docs).mkdir()
for py in here.rglob("*.py"):
    if len(py.parts) > 2:
        directory = Path(docs, *py.parts[1:-1])
        directory.mkdir(parents=True, exist_ok=True)
    file = Path(docs, *py.parts[1:-1], py.stem + ".md")
    module = ".".join([package, *py.parts[1:-1], py.stem])
    file.touch()
    file.write_text(f"::: {module}")
    tree[".".join([package, *py.parts[1:-1]])].append({module: str(file)})

print(yaml.dump(dict(tree)))

