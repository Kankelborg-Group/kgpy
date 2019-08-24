from setuptools import setup, find_packages

setup(
    name='kgpy',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'kgpy @ git+https://titan.ssel.montana.edu/gitlab/Kankelborg-Group/kgpy.git#egg=kgpy'
        'numpy'
        'matplotlib'
        'scipy'
    ],
    url='https://titan.ssel.montana.edu/gitlab/Kankelborg-Group/kgpy',
    license='',
    author='Roy Smart',
    author_email='roytsmart@gmail.com',
    description='Repository for the Full-disk Ultraviolet Rocket Spectrometer'
)
