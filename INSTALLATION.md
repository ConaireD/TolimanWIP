Hi there, the `toliman` package is not particularly complex, but the
development environment uses lots of modern tools. As a result it not 
unlikely that the most difficult part of using `toliman` will be setting 
it up. This guide should walk you through how to set up `toliman` and 
also provides basic guides to using the development tools. We are developing 
`toliman` using `git` integrated to `github` via `gh`, the `github` command
line interface (CLI). `toliman` is obviously a `python` package, and we 
recommend using the `anaconda` distribution. `poetry` is a dependancy 
managment tool for package development that behaves a lot like `cargo` for 
those who are familiar. We have used `poetry` to develop `toliman`. Our 
tests are written using the `pytest` framework, although hopefully, you 
will only need to run our tests and not edit them. That depends on how
well we have done our job. Finally, we render our documentation, which
you are currently reading using `mkdocs`. 

## Git
As a developer you have most likely heard of `git`. It is a version control
system, which allows you to revist earlier revisions of the package via a 
`commit` system. In short, when you make changes to a file the changes are 
tracked by line so that they can be undone and re-applied at will. As well as 
the core `commit` feature, I described earlier it also allows for prototyping 
outside the `main` distribution of the package via `branch`es. 

### Installing Git 
`git` can be downloaded from their [website](https://git-scm.com/downloads).
However, if you are using a unix based operating system (MacOs/Linux), you 
can install `git` using your package manager, be it `brew` or `apt/apt-get`.
You can verify that `git` is installed using `git version`.

### Using Git
`git` was originally designed to be a tool for making version control 
systems. As a result, `git` commands are divided into two layers; porcelain 
commands and plumbing commands. Almost never will you need to use a 
plumbing command, as these are the commands that were designed for implementing 
version control systems. The porcelain commands represent the `git` version 
control system, which the user normally interacts with. 

There are less than ten `git` commands that you will use in your regular 
workflow. Primarily, you will use `git add` and `git commit`. This pair of 
commands is used to create a "new version" by means of a `commit`. Once,
you have changed a file, or a set of files `stage` them using `git add 
path/to/file` (or `git add .`) to `stage` all the changes in the current 
directory. The `git status` command can be used to view what files are 
changed and staged. `git diff path/to/file` can be used to view the un`commit`ed
changes to a file (or `git diff` for all un`commit`ed changes). Once you 
have `stage`d changes you can `commit` them using `git commit`. A commit is 
normally associated with a message to explain the purpose of the commit. 
This forces you to plan your development in small chunks. An ideal `commit`
message should be a simple sentence; for example, `git commit -m "Updating
the installation guide"`. This is obviously an idealisation and by no means
what actually happens. 

Once you have a minimum viable product, it makes sense to prototype new 
features without changing the `main` version of the code. `git` provides 
the `branch` feature to facilate this kind of development. There is ongoing
debate about the validity of so called "continuous integration" workflows
in AGILE development, but I'll leave such naval gazing to the pros. To 
create a new `branch` use `git branch name`. While there is no enforced 
conventions I'm aware of for naming `branch`es, I like to use descriptive,
lower case, and hyphon separated names. For example, `git branch 
installion-guide`. To switch branches use either `git checkout` or `git switch`
both taking the `branch` name as an argument. `git checkout` is a more general
command, so I recommend using `git switch`. `branch`es can be made from any 
`branch`, not just `main`. Once you are happy with the changes on a `branch` 
you can integrate into another target `branch` using `git switch target` then 
`git merge development`, where `development` is the `branch` you were developing
on.

On paper `git` sounds like a very linear tool, meant to guarantee saftey. 
However, in the **wild** it is possible for very many curve balls to arise. 
Not the least of these is `merge` conflicts. These arise when two different 
versions of the history of a file both contain changes to the same line. 
The simplest way to produce a merge conflict is via the following `bash` 
script.

??? example

    ```bash
    (home) user@User-HP ~/Documents$ mkdir gmc
    (home) user@User-HP ~/Documents$ cd gmc
    (home) user@User-HP ~/Documents/gmc$ git init .
    (home) user@User-HP ~/Documents/gmc$ touch hello.txt
    (home) user@User-HP ~/Documents/gmc$ git add hello.txt
    (home) user@User-HP ~/Documents/gmc$ git commit -m "Creating \`hello.txt\` to demonstrate a \`merge\` commit."
    (home) user@User-HP ~/Documents/gmc$ git branch
    * main
    (home) ~/Documents/gmc$ git branch conflict
    (home) ~/Documents/gmc$ git branch 
    * main
    conflict
    (home) ~/Documents/gmc$ echo "Hello world!" > hello.txt
    (home) ~/Documents/gmc$ git add hello.txt
    (home) ~/Documents/gmc$ git commit -m "Creating one version of \`hello.txt\`." 
    (home) ~/Documents/gmc$ git switch conflict
    (home) ~/Documents/gmc$ git branch
    main
    * conflict
    (home) ~/Documents/gmc$ echo "Goodbye world!" > hello.txt
    (home) ~/Documents/gmc$ git add hello.txt
    (home) ~/Documents/gmc$ git commit -m "Creating another version of \`hello.txt\`."
    (home) ~/Documents/gmc$ git switch main
    (home) ~/Documents/gmc$ git branch 
    * main
    conflict
    (home) ~/Documents/git-merge-conflict$ git merge conflict
    Auto-merging hello.txt
    CONFLICT (content): Merge conflict in hello.txt
    Automatic merge failed; fix conflicts and then commit the result.
    (home) ~/Documents/gmc$ git diff hello.txt 
    diff --cc hello.txt
    index cd08755,7713dc9..0000000
    --- a/hello.txt
    +++ b/hello.txt
    @@@ -1,1 -1,1 +1,5 @@@
    ++<<<<<<< HEAD
     +Hello world!
    ++=======
    + Goodbye world!
    ++>>>>>>> conflict
    ```

The `<<<<<<< HEAD`, `=======` and `>>>>>> conflict` are automatically 
inserted into the file by `git`. `HEAD` is an internal pointer that 
references the current `branch`, in this case `main`. `conflict` is the 
`branch` that we are trying to `merge`. The equals signs divide the two 
different versions of the changed contents of the file. Now we would 
have to manually chose, what version we wanted to keep `conflict` of 
`main` and then `commit` that version to the history. Alternatively you 
can run `git merge --abort` to abandon the `merge` and fix up the `branch`es
individually.

We also need to know how to view and access our `commit` history. The 
best way to view it is using `git log`, which can take and `-n` argument
to show only the most recent `n` `commit`s. If you want to see what changes
are in a `commit` then you can use `git show commit_hash`. `commit` hashes 
are tricky to work with since they are long and complex. It is often much 
easier to reference a commit by how far away it is from the current `HEAD`.
For example, `git show HEAD~n` will show a `commit` `n` away from the most
recent `commit`. To `revert` a commit (undo the changes) you can use `git 
revert commit_hash`. This will automatically generate a commit message, but
you may edit it if you wish. `git reset --soft HEAD~` will undo the most 
recent commit keeping the changes available to stage and commit later. 
`git restore` can be used to unstage changes. 

While `branch`ing and `commit`ing are by far the most important tools to 
understand when using `git`, there are many other features that can be 
explored. 

### Plugins 
`git` is a very widely used tool and as a result developers and businesses 
have produced tools that plug into `git` adding additional features and/or 
making it easier to use. We use two different plugins in our development 
stratergy: `git filter-repo` and `git lfs`. Both of these can be used to 
reduce disk usage by changing the way that `git` deals with large files. 

#### Git Filter Repo
While we use `git filter-repo` to manage large files, it is a very versatile
tool that is primarily for disk-space optimisation allowing you to re-write 
`git` histories. In our case we use `git filter-repo` to write large files 
out of the `git` history. To install `git filter-repo` you can use your 
package manager provided. If this does not work the `git filter-repo` 
[website](https://github.com/newren/git-filter-repo/blob/main/INSTALL.md)
provides a detailed guide. You will need to have both `git` and `python` 
installed before you can install `git filter-repo`. 

There is a high chance you will not need `git filter-repo` in your development.
However, it is handy to know how to use when dealing with `git` in scientific 
settings. The basic usage is `git filter-repo --invert-paths --path 
I/want/to/remove/this/file/from/the/history`. The makers of `git filter-repo`
chose to specify the paths to keep rather than those to remove. This is why 
`--invert-paths` is necessary in the former snippet. I was surprised when I 
first used this plugin because it also **deletes the file** not just the 
files history. I would recommend using `git filter-repo --dry-run` before
executing your commands as a saftey measure. Since the history is changed
you cannot `revert` to the same earlier state.

Earlier I discussed `git merge` conflicts, what caused them and how to 
resolve them. When using `git filter-repo` `merge` conflicts cannot arrise
because internally `git` sees two entirely different histories. As a result
if you use `git filter-repo` to change the history of a `branch` any merges
of that `branch` with other `branch`es must be forced using `git merge --force` 
or `git merge -f`. It is usually best to use `git filter-repo` on a resh 
`branch`, check it worked and then immediately `merge` it into `main`. 
Otherwise, `merge` conflicts that might have been important can be missed.

??? example

    ```bash
    (home) user@User-HP ~/Documents$ mkdir gfr
    (home) user@User-HP ~/Documents$ cd gfr
    (home) user@User-HP ~/Documents/gfr$ git init .
    (home) user@User-HP ~/Documents/gfr$ touch big_file.txt
    (home) user@User-HP ~/Documents/gfr$ git add big_file.txt
    (home) user@User-HP ~/Documents/gfr$ git commit -m "Tracking a big file."
    (home) user@User-HP ~/Documents/gfr$ for i in {1..10000}; do echo "Hello world!" > big_file.txt; done;
    (home) user@User-HP ~/Documents/gfr$ du -h big_file.txt
    128K  big_file.txt
    (home) user@User-HP ~/Documents/gfr$ git add big_file.txt
    (home) user@User-HP ~/Documents/gfr$ git commit -m "Oooof."
    (home) user@User-HP ~/Documents/gfr$ git branch 
    * main
    (home) user@User-HP ~/Documents/gfr$ git filter-repo --path big_file.txt --invert-paths
    (home) user@User-HP ~/Documents/gfr$ git log 
    fatal: your current branch 'main' does not have any commits yet
    ```

#### Git Large File Storage 
`git lfs` is a plugin that aims to make working with large file much easier.
There [documentation](https://git-lfs.com/) is very good and can be accessed
using `git lfs --help`. In summary, `git lfs` implements a specific type of 
`git` object designed to reduce the amount of disk space that `git` uses to 
store a large file. `git` normally uses `blob`s for files but a file tracked 
using `git lfs` stores the file in this special way. 

To install `git lfs` you may use your package manager or download the binary 
from the [website](https://git-lfs.com/). To use `git lfs` first run `git 
lfs install` and then `git lfs track path/to/big/file`. You will notice that
this process has produced a new file `.gitattributes` which `git lfs` uses 
to identify what files it is tracking. From now on `git` can be used in the
normal way.

### Resources 
- [git](https://git-scm.com/)
- [bitbucket](https://www.atlassian.com/git)
- [git filter-repo](https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html)
- [git lfs](https://git-lfs.com/)

## The Github CLI 
`github` is one of many services that provide servers to store `git` 
repostories. To the best of my knowledge it is the largest such service
and it also provides a number of useful tools. For example, actions, issues
and reviews. The `toliman` repository is stored on `github` and can be 
installed using `git clone https://github.com/ConaireD/TolimanWIP`. 
I recommend installing the `github` CLI (`gh`), because it compliments 
a terminal/IDE + terminal workflow. `gh` can be installed using your 
package manager. 

### Using Gh 
`gh` is a very modular tool consisting of many command and subcommand 
patterns of the form `gh command subcommand --option`. Moreover, `gh`
is interactive and will prompt you to enter input in a text editor if
you do not provide it as an option. `gh` is a very complex tool but in 
general it is most useful for managing `issue`s and pull requests (`pr`s).

??? example

    ```bash
    (home) user@User-HP ~/Documents/toliman$ gh issue create --title "Implement an Installation Guide" --body "Hi all,
    >As the project nears completion we should make sure that it is easy to use. I think that we 
    >should provide a detailed guide on how to set up the project. Not just the basic steps but 
    >also a little bit of detail explaining how to use the various tools. This kind of information 
    >sharing is crucial to good teamwork.
    >Regards
    >Jordan" --assignee @me --label documentation
    ```

`gh issue create` tells `gh` that we are managing our `issue`s and we 
want to `create` a new one. `--title` and `--body` are self explanatory.
`--assignee` tells `gh` who to assign the issue to, in this case me (using 
the shortcut `@me`). If I wanted to assign someone else to the issue I would
type out their `github` username in full (with no `@`). For example, 
`--assignee JohnTheBaconatorOfChristopherColumbus`. Fortunately most 
usernames are much shorter than my example. 

To view an `issue` you must first know its number. To get the number run 
`gh issue list` which will list all the open issues (including their numbers).
`gh issue list` can be used to search via `--search` and can view `issue`s 
that are closed via `--state closed`. Once you have the number you can 
use `gh issue view number --comments` to view the entire conversation in the 
terminal. If you just want a summary, removing the `--comments` flag will 
just show the first and the last comment. `gh label` can be used to delete, 
edit and create labels.

### Plugins 
#### Gh Changelog
`gh` has many plugins. We only use on `gh changelog` as it allows us to 
easily keep a changelog across our versions. There is some nuance to
using this plugin. It uses `git tag`s to find the versions which are 
entered into the changelog. The canges are pulled from the `pr`s that 
have occured in between versions, and rely on the `labels` assigned to
those `pr`s. However, the tags need to be on `github` so when `push`ing
changes make sure to use the `--follow-tags` flag. To create the changelog
just run `gh changelog new` and to view it run `gh changelog view`.

### Resources 
## Python 
I imagine that you are familiar with `python`, but for the sake of
completeness and consistency it is a dynamicly typed, interpretted-bytecode 
language with inbuilt support for functional and object oriented programming. 
At present `python` does not ship with a `jit` runtime for `python` bytecode, 
but this is scheduled for the `python3.12` release. `python` is praised for 
its readability and critized for its speed. Since speed is a necessary evil 
in our `toliman` package we are using `dLux`, which in turn used `jax`, a 
third party `python/numpy` compiler and sutomatic differentiation framework.
At the `toliman` level you will rarely need to interact with `jax` directly.

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

??? example:

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

## Pytest 
Code validity is very important in modern software development. Modern 
programming languages are increasingly incorporating features to make 
programming safer and more provable. For example, `SPARK` a subset of 
`Ada` implements a contracts to guarantee compile time saftey. `rust`
is another example, implementing a borrowing API for sharing information 
that makes it almost impossible to get memory leaks. There is also 
a steady shift towards the functional programming paradigm, which, among 
other things, minimises side effects. 

??? note "Side Effects"
    A side effect is a change in the state of a program that is not 
    explicitly returned. A program cannot do anything without side 
    effects but they are a common cause of runtime errors.

`python`, being an older language, does not implement many of these 
more modern features. As a result we have to check the validity of our
code the old fashioned way: *rigorous testing*. Like most languages 
`python` has several third party implementations of testing frameworks. 
For the `toliman` project, we chose `pytest`. 

??? note "Why `pytest`"
    A common naming convention for testing frameworks is `xUnit`, where
    `x` specifies the programming language. For example, `jUnit` for 
    `java`. These pacakges usually implement a setup-teardown design.
    You program a function to setup the test/tests and one to tear it
    down. This can quickly become too rigid, preventing you from using 
    multiple setup functions. `pytest` implements a different interface
    inspired by the percieved failings of the `xUnit` ecosystems.

`pytest` is a very flexible framework. It can be invoked using `pytest`
and will automatically detect tests based on file and function/class names.
Alternatively you can run it on a specific file using `pytest path/to/file`.
It allows you to single out test classes and even test functions using 
`pytest path/to/file::TestClass::test_function`. If that was not flexible 
enough you can run tests using a keyword search via the `-k` flag. 

A `fixture` is a `pytest` artifact used to setup and teardown tests. 
`fixture`s are flexible since they can be requested by any other `fixture` 
or any other test. This modularity can make testing code much simpler. 
However, `fixture`s are not treated like a regular function. Instead 
they are executed once and cached when they are requested. In a world 
devoid of side effects this would be fine, but when they become necessary 
this can cause problems. `fixture`s are executed in the order that they
are requested so be careful.

!!! note
    `pytest` leverages the `assert` keyword to determine if a test passes. A
    test can fail and assertion or it can fail because there was an error. 
    Tests that use `assert` are more useful for checking the code is correct.

??? example "Fixtures in the Wild"
    
    ```python
    import pytest

    @pytest.fixture
    def letters_of_the_alphabet() -> list:
        return [chr(i) for i in range(97, 123)]

    def test_str_upper_on_lower_case(letters_of_the_alphabet: list) -> None:
        for char in letters_of_alphabet:
            assert char.upper().isupper()
    ```

    This test is would evaluate as 
    `test_str_upper_on_lower_case(letter_of_the_world())`. In fact, we 
    can use another `pytest` feature to improve this test. In general 
    it is advised that every unit test should test a single case, we are 
    testing twenty-six. `pytest.mark.parametrize` is similar to the inbuilt
    `map` function, but creates unique tests for each case. Using the example
    from earlier,

    ```python
    import pytest

    @pytest.mark.parametrize("char", [chr(i) for i in range(97, 123)])
    def test_str_upper_on_lower_case(char: chr) -> None:
        assert char.upper().isupper()
    ```

If you on the ball you may have noticed that all `fixture`s are evaluated 
before the test is executed. This can lead to problems when tests involve 
side effects. As a result using `yield` in a `fixture` will cause the 
code after `yield` to evaluate once the test has finished. We have used 
this feature extensively in the development of `toliman`, since we deal 
with a lot of file operations. 

??? example "`yield` in `fixture`s"
    ```python 
    import pytest

    @pytest.fixture
    def print_info() -> None:
        print(">>>")
        yield
        print("<<<")

    def test(print_info: None) -> None:
        print("TEST")
    ```
    Will output 
    ```
    >>>
    TEST
    <<<
    ```

??? tip "Writing Unit Tests"
    You should not have to write many unit tests for `toliman`, since they 
    have been programmed already. If you do have to write some unit tests 
    however, there are two guidlines that are helpful to remember: a) 
    FIRST and b) AAAC. FIRST lists the features of a good unit test and
    AAAC explains how one may implement such a test. 

    **FIRST**
    - **F**ast: Typically there are many more tests than functions. Especially 
                if you each test is specific to a single case. In a standard 
                workflow the tests are run to validate any new changes. If
                each test takes a macropspic amount of time, no one will use 
                the tests and they become redundant.
    - **I**ndependent: Each test should be a single *unit* (its in the name).
                       This way the success or failure of a test does not 
                       depend on the other tests.
    - **R**epeatable: Nothing is more frustrating than getting different 
                      results on different machines. Ideally, tests should 
                      not rely on system specific code, making debugging in 
                      a mixed team easier.
    - **S**elf-validating: This just means that the tests should pass or fail 
                           without the tester having to check. This sounds 
                           silly, but for complex outputs it can be difficult
                           to automatically determine if they are correct. 
    - **T**horough: This should be obvious, tests are not that useful if 
                    they only catch some of the bugs. Again, implementing 
                    thorough tests is time consuming and can often feal
                    very unrewarding.

    **AAAC**
    - **A**rrange: Set up the evironment of the test.
    - **A**ct: Perform the test action.
    - **A**ssert: Check if the test action produced the correct output.
    - **C**lean Up: Revert any side effects or free memory.
 
### Plugins 
`pytest` is popular, so users have created plugins to enhance the functionality
as they needed. We use several plugins for `toliman` but **none** are strictly
necessary. 

#### Pytest Xdist 
`pytest-xdist` lets us run our tests across multiple processes.
This plugin helps us make our unit tests fast (**F**IRST). Some of the 
programs we are testing are necessarily time consuming so running them
in parallel is very useful. Unfortunately it is difficult to use this 
plugin when the tests have side effects. In general side effects make
test co-dependent (F**I**RST). To invoke tests in a parallel way use 
`pytest ... -n num_processes`. The only tests in the `toliman` test 
suite that can be run on separate processes are the tests in 
`tests/test_toliman.py`.

#### Pytest Timer 
`pytest-timer` is a reporting tool. It tells us how long each test took.
This helps with **F**IRST by letting us identify which tests/functions 
need to be optimised or isolated. Once installed the plugin justs 
adds a table to the `pytest` ouput and you are not required to interact 
with it in any way. 

!!! tip 
    Some things are going to be slow. Not every program can be written so 
    that it executes in a fraction of a second. We can identify tests 
    that are slow using `pytest.mark.slow`. We can then leave these tests
    out when running pytest using `pytest ... -m "not slow"`. 

#### Pytest Cov 
`pytest-cov` addresses FIRS**T**. It shows us how much of our code is 
tested. Like `pytest-timer` it is a diagnostics tool. To generate a 
coverage report add the `--cov` flag to your `pytest` call. This will 
be ouput in the terminal giving a percentage by file. If you wish to 
see exactly where is not tested it can be used to generate a `html` 
report showing what lines were and were not tested.

#### Pytest Sugar 
While most of the plugins that we recommend have some functional value,
`pytest-sugar` purely aesthetic. This plugin beautifies the ouput of 
`pytest` and runs automatically once installed. I highly recommend it
if you plan to be running the tests much.

### Resources
- [https://docs.pytest.org/en/7.1.x/contents.html](https://docs.pytest.org/en/7.1.x/contents.html)
- [https://pytest-xdist.readthedocs.io/en/latest/](https://pytest-xdist.readthedocs.io/en/latest/)
- [https://pytest-cov.readthedocs.io/en/latest/](https://pytest-cov.readthedocs.io/en/latest/)
- [https://pypi.org/project/pytest-timer/](https://pypi.org/project/pytest-timer/)
- [https://pypi.org/project/pytest-sugar/](https://pypi.org/project/pytest-sugar/)

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

