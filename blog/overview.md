Hi there, the `toliman` package is not particularly complex, but the
development environment uses lots of modern tools. As a result it not 
unlikely that the most difficult part of using `toliman` will be setting 
it up. This guide should walk you through how to set up `toliman` and 
also provides basic guides to using the development tools. In particular
we focus on `python`, `poetry`, `anaconda`, `pytest`, `mkdocs`, `git` and
`gh`.

!!! note
    These resources are not intended to be complete API guides. Instead they 
    focus on the practical use cases from `toliman`. A user who is not 
    familiar with the tools should be able to skim the relevant sections of 
    this blog and gain enough familiarity to start development. This blog 
    should also be useful as a reference for veterans.

Many of these tools are not necessary to start developing, but `git`, `python` 
and `poetry` are required to install `toliman`. `pytest` is required to 
use the tests and `mkdocs` is required to view the documentation locally. 
`gh` is only mentioned as a productivity enhancer and `anaconda` is useful
if you use `python` for other projects. If you recognise any of these 
tools, you may realise they primarily benefit a terminal workflow. If you 
prefer a GUI, `gh` is not recommended.

## Installing `toliman`
!!! tip
    If you do not have a package manager installed, I highly recommend 
    downloading one. Whilst everything is possible to install via archive 
    downloads, unless you are familiar with how to do this on your operating 
    system. I do not recommend doing it this way. On `MacOS`, `homebrew` is 
    the recommended package manager. It can be installed using,

    ```
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

Throughout this blog I am going to adopt some conventions when it comes to 
terminal demonstrations. Each command will be prefaced by 
`(home) user@Users-HP ~path/to/pwd$`. `(home)` refers to the active virtual
environment. `user@Users-HP` is just some standard meta-information that 
can be ignored and `~path/to/pwd$` indicates the current directory.
Assuming you have a package manager the installation should be as simple as:
```bash
(home) user@Users-HP ~$ sudo apt install python
(home) user@Users-HP ~$ sudo apt install git
(home) user@Users-HP ~$ sudo apt install poetry 
```

!!! tip 
    You can check *often* if a package has installed using the `--version` tag. 
    For example, `git --version`. 

Once you have `git` installed, you can clone the source code for the `toliman` 
forwards model. 
```
(home) user@Users-HP ~$ git clone https://github.com/ConaireD/TolimanWIP.git
```
This will clone the repository into `TolimanWIP`, if you want to use a 
different name, pass the desired name as a second argument. For example,
```
(home) user@Users-HP ~$ git clone https://github.com/ConaireD/TolimanWIP.git toliman-fw
```

!!! tip
    Depending on how you plan to use `toliman`, it may not make sense for you 
    to waste disk space keeping the entrie `git` history. Passing the `--depth`
    tag, allows you to specify the number of commits to clone. If space is a 
    problem, I recommend using `--depth 1`, as this will just keep the current 
    state.

Now that the `toliman` source code is installed, we can install the package. 
Making sure that `poetry` is installed it is as simple as `poetry install`. 
This will some time to finish running, because the model has a very large 
number of dependencies. 

!!! note
    Normally, I like to try keep the number of dependencies that my projects 
    have as small as possible. However, `dLux` requires `matplotlib` and 
    `jupyter`, which both come with a vast number of dependencies. In the 
    end the large number of dependencies is why we use `poetry`.

## Using `conda`
Anaconda is a popular distibution of `python` that comes equipped with 
a virtual environment manager. If you wish to install `toliman` using 
Anaconda there are a few extra steps.
```bash
(home) user@Users-HP ~$ brew install python
(home) user@Users-HP ~$ brew install git
(home) user@Users-HP ~$ brew install poetry 
(home) user@Users-HP ~$ brew install anaconda 
```
We will want to create a new virtual environment for `toliman`. This can 
be done using:
```bash
(home) user@Users-HP ~$ conda create --name tlmn python==3.10.8
(home) user@Users-HP ~$ conda activate tlmn
(tlmn) user@Users-HP ~$
```
From here everything proceeds the same. 

## `python==3.11`
If you want to use `python==3.11` there are some extra steps again. 
