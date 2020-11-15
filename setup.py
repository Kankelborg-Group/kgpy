from setuptools import setup, find_packages

setup(
    name='kgpy',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'astropy',
        'sunpy[all]',
        'pandas',
        # 'beautifultable',
        # 'pywin32 ; platform_system=="Windows"',
        'pytest',
        'wget',
        'urlpath',
        'shapely',
        # 'pythonocc-core',
    ],
    url='https://titan.ssel.montana.edu/gitlab/Kankelborg-Group/kgpy',
    license='',
    author='Roy Smart',
    author_email='roytsmart@gmail.com',
    description='Repository for the Kankelborg Group\'s common codebase'
)
