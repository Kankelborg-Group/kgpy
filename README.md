# kgpy

[![tests](https://github.com/Kankelborg-Group/kgpy/workflows/tests/badge.svg)](https://github.com/Kankelborg-Group/kgpy/actions?query=workflow%3Atests)
[![Documentation Status](https://readthedocs.org/projects/kgpy/badge/?version=latest)](https://readthedocs.org/projects/kgpy/builds/)

This library aims to contain general code that does not belong with a specific Kankelborg Group project.

## Documentation

The [documentation](https://kgpy.readthedocs.io/) is currently hosted by Read the Docs.
It is automatically compiled every time a commit is pushed to `master`.

## Installation

This project requires `conda` for all functionality.
Make sure that `conda` is installed on your system before continuing.

We think that most users of this project should install it in developer mode.

```shell script
git clone https://titan.ssel.montana.edu/gitlab/Kankelborg-Group/kgpy.git
conda install conda-build
conda develop kgpy
```
