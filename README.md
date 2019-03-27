# kgpy

Software libraries shared between Kankelborg-Group projects.

## Installation

### Python Dependencies
Shapely needs to be installed using the wheel file on Windows.
Download the file [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely), and install using:
```
python -m pip install Shapely-1.6.4.post1-cp37-cp37m-win_amd64.whl
```

### PGI Compiler
This software relies on openACC, which is currently best supported by the PGI compiler.
The community version of this compiler can be downloaded from [here](https://www.pgroup.com/products/community.htm]), and installed using the following commands.
```
cd ~/Downloads
tar -xzvf pgilinux-2018-184-x86-64.tar.gz
sudo ./install_components/install
```
Now to access the installation, the following lines should be added to your `.bashrc` file.
```
$ export PGI=/opt/pgi;
$ export PATH=/opt/pgi/linux86-64/18.4/bin:$PATH;
$ export MANPATH=$MANPATH:/opt/pgi/linux86-64/18.4/man;
$ export LM_LICENSE_FILE=$LM_LICENSE_FILE:/opt/pgi/license.dat; 
```
To use the eclipse PGI plugin, a copy can be found on Roy Smart's gitlab page on the Titan sever.
To install:
```
cd ~/Downloads
git clone https://titan.ssel.montana.edu/gitlab/roys/PGI-Eclipse-plugin.git
cd PGI-Eclipse-plugin
sudo ./install
```
The PGI compiler toolchain should now be ready to use.
