from setuptools import setup, find_packages

setup(
    name='kgpy',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'pytest',
        'numpy',
        'matplotlib',
        'scipy',
        'astropy',
        'sunpy[all]',
        'aiapy',
        'pandas',
        'wget',
        'urlpath',
        'shapely',
        'sphinx-autodoc-typehints',
        'astropy-sphinx-theme',
    ],
    url='https://titan.ssel.montana.edu/gitlab/Kankelborg-Group/kgpy',
    license='',
    author='Roy Smart',
    author_email='roytsmart@gmail.com',
    description='Repository for the Kankelborg Group\'s common codebase'
)
