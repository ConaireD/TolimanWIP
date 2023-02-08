## Poetry  
Dependencies are third party libraries used by a package. Quite often 
dependancies contribute to software bloat, because although the entire 
library needs to be installed you may only use/need a small subset. As
a goal of modern development is modularity. Every dependency tries to do 
only one thing (OK not quite **one** thing), with a clear purpose. This 
creates its own problems, since now many small dependencies are needed 
which in turn may depend on each other or yet more dependencies. Depending
on whether or not dependcies are actively maintained, your dependencies 
can come to rely upon different versions often within some window. You 
can imagine that for any sizeable project downloading compatible versions
of the dependencies can become an ourtight nightmare. 

`pip` the default `python` package manager has some very basic dependancy 
management functionality, but for the most part it is *superficial*; printed 
warnings etc. Although `toliman` is a small package by software standards 
however, `jax`, which is a dependancy of `dLux` is under rapid development
publishing many new versions over the lifetime of the project. This is one
way that dependencies can become incompatible, because changes in the `jax`
API may break packages that use it (for example `dLux`). It is common when 
developing `python` projects to use `pip` since it is not a dependency and 
provides most of the necessary functionality. However, due to our past 
experiences with `jax` we chose to use the more modern tool `poetry`.

Most modern programming languages come with a dependancy management tool,
for `python` this is `poetry`. `poetry`, as I aluded to in the paragraph 
above is ironically a third party tool and hence a dependency itself. 
By and large dependency management tools are able to automatically select 
the correct versions of packages too install preventing this type of 
development headache. Moreover, `poetry` also simplifies other common 
processes such as `install`ing, `build`ing and `publish`ing. When working 
with `toliman`, you are unlikely to require much familiarity with `poetry` 
as most/all of the dependencies are already established.

### Installing Poetry  
Unfortunately, `poetry` cannot yet be installed using a package manager 
so you will need to download it from the internet. On MacOS/Linux `poetry` 
can be installed using `curl -sSL https://install.python-poetry.org | python3 -`.


### Using Poetry  
This guide is very cursory, since it is unlikely you will need to use 
`poetry` much in your development journey with toliman. The first useful 
command is `poetry show package`. This will print useful information about
the package you selected. In fact, it will even print in **color!** Let's
imagine that you have stumbled across a new package that you think `toliman`
will benfit from. You decide to use it. To register it as a dependency run 
`poetry add package` and `poetry` will automatically find a version that is 
compatible with the existing versions and install it. How does `poetry` work? 
You may notice the `pyproject.toml` file in the root directory of `toliman`. 
This is the file that `poetry` uses to manage/track the dependencies of 
`toliman`. 

??? aside

    `python` dependency management has incrementally evolved in the thirty 
    odd years of the languages lifetime. I'm not sure of the early years 
    but after some time the `setuptools` package was created in the standard 
    library, which could be used to specify a `build` within a `python` script
    called `setup.py`. While this allowed for some powerful metaprogramming for
    experienced users, in general the interface was messy. Somewhere around 
    this time `build` tools for other languages were increasingly handling 
    dependencies. For example, `ant` inspired `maven` in the `java` ecosystem,
    which is both a build tool and full blown dependency manager. Over this 
    period a very large number of markup languages were created, providing 
    well defined syntaxes. Build tools and dependency management tools 
    increasingly started to leverage markup languages instead of implementing 
    domain specific languages for their purposes. Eventually, `rust` released 
    along with a `cargo`, a dependency management and build tool. 
    `cargo` used the `toml` markup language as the interface to
    its dependency specification. Furthermore, `cargo` also took
    things one step further and automated many processes, so the typical 
    user never interacted with the `cargo.toml`. The `poetry`
    interface is very similar to the `cargo` one.

To install the dependencies for `toliman` simply run `poetry install`. This
will take a while so make sure you have a coffee and a good book nearby.
By default this will install `toliman` in development mode. This just means 
that instead of placing a `.whl` file in the `conda` environment, `poetry` 
has placed a file pointing to the code itself. This is very handy because 
it means that any changes you make instantly become effective even when 
using the installed version. To fully install `toliman` run `poetry build`.
This will create a `dist` directory, containing two files. One will be a 
`.whl`, the other a `.tar.gz`. We only care about the the `.whl`. To install 
the `.whl` use `pip install dist/name.whl`.

`poetry` facilitates the grouping of dependencies. This can be very useful,
but is often unecessary. For example, imagine you are developing a package
that groups its dependencies by `tests`, `core`, and `documentation`. You 
may just be using the package and do not plan to modify the code or 
documentation. In this case `poetry install --group core` would install 
only the dependencies required to use the package, saving disk space 
and a little time. You can add dependencies to a new group using `poetry 
add --group group-name`. 

Say you are using several dependencies to manage your documentation. As 
you learn more about these dependencies you discover there is a lot of 
overlapping functionality between them. In the end you discover that 
you can acheive the same results using half the number of dependencies.
`poetry` makes removing dependencies as easy as `poetry remove dependency`
which will uninstall dependency and remove it from the `pyproject.toml`.

#### Resources  
- [https://python-poetry.org/docs/](https://python-poetry.org/docs/)

