# Contributing
--------------
The `toliman` package is managed using `poetry` and it is helpful if you also 
have `conda` installed. Firstly check you `python` version, which can be done 
using by typing `python --version` in the command prompt. You will need to have
`python >= 3.7` installed but it is recommended that you use `python == 3.10`. 
The package development was done using `python == 3.10.8`, which was the 
highest supported `python` for `conda` at the time of development. 

Once you have checked your `python` installation is up to date, we can install 
`poetry`. If you are using `conda` then I would recommend completing all these 
steps in a fresh `conda` environment (you can create a new environment using
`conda create -n your-name-here`). The `poetry` 
[documentation](https://python-poetry.org/docs/) explains how to install 
`poetry` which needs to be done using the system equivalent of `curl`. 
Check `poetry` is installed by running `poetry --version`. The package was 
developed using `poetry == 1.3.2`. 

Now that `poetry` is installed you can use `git` to clone the repository. If 
you do not have `git` installed, detailed instructions can be found in the 
`git` [documentation](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
As with `poetry` and `python` you can check the version of `git` that is 
installed using `git --version`. Once `git` is installed run `git clone
https://github.com/ConaireD/TolimanWIP` to import the repository onto your 
machine. 

This process should have cloned the project into a new folder called 
`TolimanWIP`. Move into that directory using `cd` and then run `poetry install`.
This will take some time as `poetry` resolves the packages dependencies. So
go grab a coffee sit back and relax while it runs (should take about 1-10min
depending on your connection). You can now make changes to the source code.
If you want these changes to join the `git` repository you will instead need
to create a `fork` (unless you have write permission). 

The full installation and setup process on Ubuntu, without `git` or `poetry`
pre-installed, but with `conda` looks like:
```bash
$ conda create -n toliman python==3.10.8
$ conda activate toliman
$ python --version
$ curl -sSL https://install.python-poetry.org | python3 -
$ poetry --version
$ sudo apt update
$ sudo apt install git
$ git version 
$ git clone https://github.com/ConaireD/TolimanWIP
$ cd TolimanWIP
$ poetry install 
```
If you are planning to change the code, then be aware that we are using `black`
as an auto-format-tool on both the tests and the source code. Before pushing make
sure that you have run the tests by invoking `pytest` and formatted the code 
using `black`. 

Externally, we have a few other recommendations. Where it makes sense to do so 
use `from package import ...` rather than `import package; package.(...)` but
this is not a strict rule and is just a guideline. To get updates from the 
repository run `git pull` and then re-install using `poetry install`. Note,
it should be much faster after the first time because `poetry` has already 
installed all the dependencies.
