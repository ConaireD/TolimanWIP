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
terminal demonstrations.
Assuming you have a package manager the installation should be as simple as:
```
(home) user@Users-HP ~$ sudo apt install git
(home) user@Users-HP ~$ git --version
git version 2.34.1
(home) user@Users-HP ~$ sudo apt install poetry 
(home)

