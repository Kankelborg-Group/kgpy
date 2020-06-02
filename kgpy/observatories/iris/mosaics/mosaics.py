import typing as typ
import pathlib
import wget
import astropy.units as u
import astropy.constants
import astropy.wcs
import numpy as np

from . import url

__all__ = ['default_path', 'download' ]

default_path = pathlib.Path(__file__).parent / 'data'


def download(path: pathlib.Path = None) -> typ.List[pathlib.Path]:
    if path is None:
        path = default_path
    path.mkdir(parents=True, exist_ok=True)
    files = []

    for mosaic_path in url.path_list:
        u = url.base / mosaic_path
        file = path / mosaic_path.name
        if not file.exists():
            wget.download(str(u), out=str(file))
        files.append(file)

    return files


def test_download(capsys):
    with capsys.disabled():
        download()
