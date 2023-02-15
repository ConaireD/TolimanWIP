## Installation
This is a minimal installation guide that does not tell you how to use any 
of the powerful tools you will install. If you want more information search 
for *yet another astronomy blog* (YAAB), a blog that was actuall spun out of
the handover for this project. If you just wish to use the forwards model and 
are not interested in developing it further you can just run `pip install
toliman`. However, installing a development copy is a little more difficult. 
Firstly, you will need to install `git` and `poetry`. Using a package 
manager (like `brew`) it is as simple as `brew install git` and `brew install 
poetry`. 

!!! tip 
    If you want to validate that these are indeed the correct pacakges 
    before installing them onto your computer most package managers support a 
    `search` function. For example `brew search poetry`. If you want to 
    validate the packages are correctly installed then run `git --version` 
    and `poetry --version`. 

Next run `git clone https://ConaireD/TolimanWIP.git` which will clone 
the repository into `TolimanWIP`. You can change the name using the `-O`
flag. Move into cloned repository and run `poetry install`. This will 
take a moment but once it is finished you will have a safe debug build. 

Unfortunately this is not the end of the road. Open a `python` terminal 
using `python` and add the following two lines,
```python 
import toliman
toliman.build.install_toliman(number_of_wavelengths = ..., force = True)
```
This should fetch a large number of data files from the internet and then 
you are good to go. The `number_of_wavelengths` parameter indicates how 
many wavelengths to use for the spectrum of Alpha Centauri.

!!! tip
    By default `toliman` will create a `.assets` folder to store the 
    datafiles in. If you want it to be something else you must export 
    the constant `TOLIMAN_HOME` before running `install_toliman`. 
    I would recommend adding the definition to you `.zprofile`/`.bashrc` 
    so that you do not need to export it everytime you want to use 
    `toliman`.

!!! note
    You may be wondering why we did not make the installation run via a 
    command line interface. The reason is that it would need to be 
    invoked via `python install.py`. While this looks very clean it does 
    not give the user very much flexibilty if they wish to update individual 
    datafiles later. The `toliman.build` submodule handles this and the 
    API is heavily documented. 
