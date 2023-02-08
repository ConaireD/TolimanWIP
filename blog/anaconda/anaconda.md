### Anaconda 
Anaconda or `conda` is a popular distribution of `python` that ships within 
a virtual environment manager. We used `conda` to develop `toliman` and 
recommend it to others who are involved on the project. A virtual environment
provides a pointer to a set of executables and packages, ensuring that once 
the environment is activated the versions it points to are used. This is most
useful when developing multiple packages, with different versions of shared
dependancies.  

#### Installing Anaconda 
To install anaconda you will need to download the installer from the 
[Anaconda website](https://docs.anaconda.com/anaconda/install/). Follow 
the installation instructions specific to your operating system from there 
onwards. On MacOS/Linux you will need to execute the downloaded `bash` 
script using `bash path/to/script` and it will do the rest for you. I 
believe that it is safe to remove the script once `conda` is installed.

#### Using Anaconda 
Imagine you are developing `toliman` which uses `python3.10.8`, and also 
developing `steampunkfairytale` which uses `python3.8`.
You can create an environment for each and switch between as needed. 

??? example

    ```bash
    (home) user@Users-HP: ~/Documents$ conda create toliman python=3.10.8 
    (home) user@Users-HP: ~/Documents$ conda activate toliman
    (toliman) user@Users-HP: ~/Documents$ cd toliman
    (toliman) user@Users-HP: ~/Documents/toliman$ echo "Developing toliman ... Done!"
    Developing toliman ... Done!
    (toliman) user@Users-HP: ~/Documents/toliman$ conda deactivate
    (home) user@Users-HP: ~/Documents/toliman$ cd ..
    (home) user@Users-HP: ~/Documents$ conda create steampunckfairytale python=3.8
    (home) user@Users-HP: ~/Documents$ conda activate steampunkfairytale 
    (steampunkfairytale) user@Users-HP: ~/Documents$ cd spft
    (steampunkfairytale) user@Users-HP: ~/Documents/spft$ echo "Developing steampunkfairytale ... Done!"
    Developing steampunkfairytale ... Done!
    (steampunkfairytale) user@Users-HP: ~/Documents/spft$ conda deativate steampunkfairytale
    (home) user@Users-HP: ~/Documents/spft$ cd ..
    (home) user@Users-HP: ~/Documents$ 
    ```

`conda` also comes with a package manager (similar to `pypi` + `pip`), which 
can be used to install packages. The interface is more or less the same 
as `pip` which is `python`s default package manager. I am assuming familiary
with `pip` but if you need more information the 
[documentation](https://pip.pypa.io/en/stable/) is very good.

#### Resouces 
- [Anaconda](https://docs.anaconda.com/anaconda)

