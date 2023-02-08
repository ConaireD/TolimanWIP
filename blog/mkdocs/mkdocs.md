
## Mkdocs
We have used `mkdocs` to generate our documentation. `mkdocs` makes it easy 
to produce high quality static documentation without too much hassle, turning 
markdown files into a website. `mkdocs` reads markdown files and configures 
the website from a `yaml` file; `mkdocs.yml`. While `mkdocs` forms the backbone
of our documentation engine we are using a number of plugins that make the 
interface entirely alien from vanilla `mkdocs`. We have chosen to use `mkdocs` 
in this way to adhere to the concept of literate programming. To this end we 
use the plugins `mkdocs-same-dir`, `mkdocs-simple` and `mkdocstrings`.

??? note "Vanilla `mkdocs` and Literate Programming"
    If you we to start developing a new package tomorrow using `poetry` 
    and `mkdocs`, you could quite simply do:

    ```bash
    (home) user@Users-HP ~/Documents$ mkdocs new mypackage && cd mypackage
    (home) user@Users-HP ~/Documents/mypackage$ poetry init .  
    ```

    Then you could type your documentation in the automatically generated 
    `docs` folder using markdow and view it using `mkdocs serve`. The 
    problem with this is that if your API changes you have to manually
    change this in the documentation. If your API is internally documented 
    using docstrings this means that your work is doubled. Literate 
    programming is about recognising that documentation is just as important
    to programming as actually writing code is. 

    In particular, one of the goals of literate programming is to provide 
    the documentation in the same place as the body of the code. Most languages
    implement this via docstrings/multiline comments, and it may be formalised 
    further by additional tools. `java` is a good example, the `java` 
    development kit containing `javadoc` a tool to automatically render 
    documentation websites from commented code. For us, `mkdocstrings` and
    `mkdocs-simple` provide the means to implement literate programming.

### Plugins 
#### Mkdocs Same Dir
`mkdocs-same-dir` let's us write our documentaton in the same directory as 
our code. simplifies the structure of the package and more closely ties the 
documentation (and its structure) to the code of the package. Once installed
`mkdocs-same-dir` is very easy to use. Opening the `mkdocs.yml` add the 
following line lines to your `mkdocs.yml`

```yaml
docs_dir: .

plugins:
- same-dir
```

#### Mkdocs Simple 
Like `mkdocs-same-dir`, `mkdocs-simple` is easy to use. It tells `mkdocs` 
not just to look for markdown files, but also to look for source files 
containing multiline comments/strings with the `md` flag. To use it 
add the following line to your `mkdocs.yml`:

```yaml
plugins:
- simple
```

??? example
    ```python 
    # src/https.py
    """md
    ## Overview
    This package interacts extensively with the internet. Due to the nature 
    of the product we enforce long timeout thresholds and high retry counts.
    To make sure that this is enforced we provide an interface (via requests)
    that is used internally. This is managed via the ..
    """
    import requests

    class HttpRequest(requests.Request):
        """
        """
    ```

    In most cases this can avoid the creation of overview files for 
    submodules. While it is handy, it is easy to overuse and I would 
    recommend caution when chosing whether or not to use it.

#### Mkdocstrings 
`mkdocstrings` is used to automatically generate documentation from docstrings.
This is extremely handy and can be combined with `mkdocs-simple` to great 
affect. When using `mkdocstrings` with vanilla `mkdocs` you would have to 
create a file in `docs/submodule/myclass.py` and add into it 
`::: src.submodule.MyClass`. This can quickly get out of hand, and you 
end up with all these practically empty markdown files. When using simple 
it can be done in place. Let's revisit my earlier example.

??? example
    ```python 
    # src/https.py
    """md
    ## Overview
    This package interacts extensively with the internet. Due to the nature 
    of the product we enforce long timeout thresholds and high retry counts.
    To make sure that this is enforced we provide an interface (via requests)
    that is used internally. This is managed via the ..

    ::: src.hhtps.HttpRequest
    """
    import requests

    class HttpRequest(requests.Request):
        """
        """
    ```

    Now `mkdocs` will not only output the overview, but also the fully 
    documented API of the `HttpRequest` class.

Unfortunately, it can be quite difficult to setup `mkdocstrings`, since
they tried to make it a more general tool, for multiple languages. As a 
result you have to specify different handlers. For `toliman` we are using 
the `mkdocstrings-python-legacy` version, since this uses `pytkdocs` as 
the backend. I chose this version because it allows documentation to 
be inherited from parent classes. To use `mkdocstrings`, add the following 
to your `mkdocs.yml`:

```yaml
plugins:
- mkdocstrings
```

#### Mkdocs Material 
Yay! We have generated some documentation in a way that adheres to the rules
of literate programming. Now we are confronted with a very severe problem. 
They are ugly and generic. Just as `pytest-sugar` was purely aesthetic 
`mkdocs-materical` is solely about improving the look and feal of the 
documentation. This is a theme for `mkdocs` and it can be configured with:

```yaml
theme:
  name: material
```

### Resources 
- [https://www.mkdocs.org/](https://www.mkdocs.org/) 
- [https://squidfunk.github.io/mkdocs-material/](https://squidfunk.github.io/mkdocs-material/)
- [https://mkdocstrings.github.io/](https://mkdocstrings.github.io/)
- [https://www.althack.dev/mkdocs-simple-plugin/v2.2.0/](https://www.althack.dev/mkdocs-simple-plugin/v2.2.0/)
- [https://oprypin.github.io/mkdocs-same-dir/](https://oprypin.github.io/mkdocs-same-dir/)

