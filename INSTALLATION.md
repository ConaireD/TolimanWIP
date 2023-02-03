# Overview <a name="overview">
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

# Git <a name="git">
As a developer you have most likely heard of `git`. It is a version control
system, which allows you to revist earlier revisions of the package via a 
`commit` system. In short, when you make changes to a file the changes are 
tracked by line so that they can be undone and re-applied at will. As well as 
the core `commit` feature, I described earlier it also allows for prototyping 
outside the `main` distribution of the package via `branch`es. 

## Installing Git <a name="git.installing">
`git` can be downloaded from their [website](https://git-scm.com/downloads).
However, if you are using a unix based operating system (MacOs/Linux), you 
can install `git` using your package manager, be it `brew` or `apt/apt-get`.
You can verify that `git` is installed using `git version`.

## Using Git <a name="git.using">
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
```bash
(home) ~/Documents$ mkdir git-merge-conflict
(home) ~/Documents$ cd git-merge-conflict
(home) ~/Documents/git-merge-conflict$ git init .
(home) ~/Documents/git-merge-conflict$ touch hello.txt
(home) ~/Documents/git-merge-conflict$ git add hello.txt
(home) ~/Documents/git-merge-conflict$ git commit -m "Creating \`hello.txt\` to demonstrate a \`merge\` commit."
(home) ~/Documents/git-merge-conflict$ git branch
main*
(home) ~/Documents/git-merge-conflict$ git branch conflict
(home) ~/Documents/git-merge-conflict$ git branch 
main*
conflict
(home) ~/Documents/git-merge-conflict$ echo "Hello world!" > hello.txt
(home) ~/Documents/git-merge-conflict$ git add hello.txt
(home) ~/Documents/git-merge-conflict$ git commit -m "Creating one version of \`hello.txt\`." 
(home) ~/Documents/git-merge-conflict$ git switch conflict
(home) ~/Documents/git-merge-conflict$ git branch
main
conflict*
(home) ~/Documents/git-merge-conflict$ echo "Goodbye world!" > hello.txt
(home) ~/Documents/git-merge-conflict$ git add hello.txt
(home) ~/Documents/git-merge-conflict$ git commit -m "Creating another version of \`hello.txt\`."
(home) ~/Documents/git-merge-conflict$ git switch main
(home) ~/Documents/git-merge-conflict$ git branch 
main*
conflict
(home) ~/Documents/git-merge-conflict$ git merge conflict
Auto-merging hello.txt
CONFLICT (content): Merge conflict in hello.txt
Automatic merge failed; fix conflicts and then commit the result.
(home) ~/Documents/git-merge-conflict$ git diff hello.txt 
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

## Plugins <a name="git.plugins">
`git` is a very widely used tool and as a result developers and businesses 
have produced tools that plug into `git` adding additional features and/or 
making it easier to use. We use two different plugins in our development 
stratergy: `git filter-repo` and `git lfs`. Both of these can be used to 
reduce disk usage by changing the way that `git` deals with large files. 

### Git Filter Repo <a name="git.plugins.git_filter_repo">
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

Below I have created a very simple example of how you might use 
`git filter-repo`:
```bash
(home) ~/Documents$ mkdir git-filter-repo
(home) ~/Documents$ cd git-filter-repo
(home) ~/Documents/git-filter-repo$ git init .
(home) ~/Documents/git-filter-repo$ touch big_file.txt
(home) ~/Documents/git-filter-repo$ git add big_file.txt
(home) ~/Documents/git-filter-repo$ git commit -m "Tracking a big file."
(home) ~/Documents/git-filter-repo$ for i in {1..10000}; do echo "Hello world!" > big_file.txt; done;
(home) ~/Documents/git-filter-repo$ du -h big_file.txt
128K  big_file.txt
(home) ~/Documents/git-filter-repo$ git add big_file.txt
(home) ~/Documents/git-filter-repo$ git commit -m "Oooof."
(home) ~/Documents/git-filter-repo$ git branch 
* main
(home) ~/Documents/git-filter-repo$ git filter-repo --path big_file.txt --invert-paths
(home) ~/Documents/git-filter-repo$ git log 
fatal: your current branch 'main' does not have any commits yet
```

### Git Large File Storage <a name="git.plugins.git_large_file_storage">
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

## Resources <a name="git.resouces"> 
- [git](https://git-scm.com/)
- [bitbucket](https://www.atlassian.com/git)
- [git filter-repo](https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html)
- [git lfs](https://git-lfs.com/)

# The Github CLI <a name="gh">
`github` is one of many services that provide servers to store `git` 
repostories. To the best of my knowledge it is the largest such service
and it also provides a number of useful tools. For example, actions, issues
and reviews. The `toliman` repository is stored on `github` and can be 
installed using `git clone https://github.com/ConaireD/TolimanWIP`. 
I recommend installing the `github` CLI (`gh`), because it compliments 
a terminal/IDE + terminal workflow. `gh` can be installed using your 
package manager. 

## Using Gh <a name="gh.using">
`gh` is a very modular tool consisting of many command and subcommand 
patterns of the form `gh command subcommand --option`. Moreover, `gh`
is interactive and will prompt you to enter input in a text editor if
you do not provide it as an option. `gh` is a very complex tool but in 
general it is most useful for managing `issue`s and pull requests (`pr`s).
Consider the following example,
```bash
(home) ~/Documents/toliman$ gh issue create --title "Implement an Installation Guide" --body "Hi all,
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
use `gh issue view # --comments` to view the entire conversation in the 
terminal. If you just want a summary, removing the `--comments` flag will 
just show the first and the last comment. `gh label` can be used to delete, 
edit and create labels.

## Plugins <a name="gh.plugins">
### Gh Changelog <a name-"gh.plugins.gh_changelog">
`gh` has many plugins. We only use on `gh changelog` as it allows us to 
easily keep a changelog across our versions. There is some nuance to
using this plugin. It uses `git tag`s to find the versions which are 
entered into the changelog. The canges are pulled from the `pr`s that 
have occured in between versions, and rely on the `labels` assigned to
those `pr`s. However, the tags need to be on `github` so when `push`ing
changes make sure to use the `--follow-tags` flag. To create the changelog
just run `gh changelog new` and to view it run `gh changelog view`.

## Resources <a name="gh.resources">
# Python <a name="python">
I imagine that you are familiar with `python`, but for the sake of
completeness and consistency it is a dynamicly typed, interpretted-bytecode 
language with inbuilt support for functional and object oriented programming. 
At present `python` does not ship with a `jit` runtime for `python` bytecode, 
but this is scheduled for the `python3.12` release. `python` is praised for 
its readability and critized for its speed. Since speed is a necessary evil 
in our `toliman` package we are using `dLux`, which in turn used `jax`, a 
third party `python/numpy` compiler and sutomatic differentiation framework.
At the `toliman` level you will rarely need to interact with `jax` directly.

## Anaconda <a name="python.anaconda"
Anaconda or `conda` is a popular distribution of `python` that ships within 
a virtual environment manager. We used `conda` to develop `toliman` and 
recommend it to others who are involved on the project. A virtual environment
provides a pointer to a set of executables and packages, ensuring that once 
the environment is activated the versions it points to are used. This is most
useful when developing multiple packages, with different versions of shared
dependancies.  

### Installing Anaconda <a name="python.anaconda.installing">
To install anaconda you will need to download the installer from the 
[Anaconda website](https://docs.anaconda.com/anaconda/install/). Follow 
the installation instructions specific to your operating system from there 
onwards. On MacOS/Linux you will need to execute the downloaded `bash` 
script using `bash path/to/script` and it will do the rest for you. I 
believe that it is safe to remove the script once `conda` is installed.

### Using Anaconda <a name="python.anaconda.using">
Imagine you are developing `toliman` which uses `python3.10.8`, and also 
developing `steampunkfairytale` which uses `python3.8`.
You can create an environment for each and switch between as needed. 

```bash
(home) user@Users-HP: ~/Documents$ conda create toliman python=3.10.8 
(home) user@Users-HP: ~/Documents$ conda activate toliman
(toliman) user@Users-HP: ~/Documents$ cd toliman
(toliman) user@Users-HP: ~/Documents/toliman$ echo "Developing toliman ... Done!"
Developing toliman ... Done!"
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

### Resouces <a name="python.anaconda.resouces">
- [Anaconda](https://docs.anaconda.com/anaconda)

## Poetry <a name="python.poetry"> 
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

### Installing Poetry <a name="python.poetry.installing"> 
Unfortunately, `poetry` cannot yet be installed using a package manager 
so you will need to download it from the internet. On MacOS/Linux `poetry` 
can be installed using `curl -sSL https://install.python-poetry.org | python3 -`.


### Using Poetry <a name="python.poetry.using"> 
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

!!! tip
  This is a tip?

<style>
  .aside {
    background-color: cyan;
    border-radius: 15px 50px;
    border: 2px solid blue;
    padding: 20px;
  }
</style>
<div class="aside">
  <h3> 
    Aside: 
  </h3>
  <p> 
    <code>python</code> dependency management has incrementally evolved in the thirty 
    odd years of the languages lifetime. I'm not sure of the early years 
    but after some time the <code>setuptools</code> package was created in the standard 
    library, which could be used to specify a <code>build</code> within a <code>python</code> script
    called `setup.py`. While this allowed for some powerful metaprogramming for
    experienced users, in general the interface was messy. Somewhere around 
    this time <code>build</code> tools for other languages were increasingly handling 
    dependencies. For example, <code>ant</code> inspired <code>maven</code> in the <code>java</code> ecosystem,
    which is both a build tool and full blown dependency manager. Over this 
    period a very large number of markup languages were created, providing 
    well defined syntaxes. Build tools and dependency management tools 
    increasingly started to leverage markup languages instead of implementing 
    domain specific languages for their purposes. Eventually, <code>rust</code> released 
    along with a <code>cargo</code>, a dependency management and build tool. 
    <code>cargo</code> used the <cargo>toml</cargo> markup language as the interface to
    its dependency specification. Furthermore, <code>cargo</code> also took
    things one step further and automated many processes, so the typical 
    user never interacted with the <code>cargo.toml</code>. The <code>poetry</code>
    interface is very similar to the <code>cargo</code> one.
  </p>
</div>

To install the dependencies for `toliman` simply run `poetry install`. This
will take a while so make sure you have a coffee and a good book nearby.


### Resources <a name="python.poetry.resources"> 
# Pytest <a name="pytest">
## Installing Pytest <a name="pytest.installing">
## Using Pytest <a name="pytest.using">
## Plugins <a name="pytest.plugins">
### Pytest Xdist <a name="pytest.plugins.pytest_xdist">
### Pytest Timer <a name="pytest.plugins.pytest_timer">
### Pytest Cov <a name="pytest.plugins.pytest_cov">
### Pytest Sugar <a name="pytest.plugins.pytest_sugar">
## Resources
# Mkdoc <a name="mkdocs">
## Installing Mkdocs <a name="mkdocs.installing">
## Using Mkdocs <a name="mkdocs.using">
## Plugins <a name="mkdocs.plugins">
### Mkdocs Same Dir <a name="mkdocs.plugins.mkdocs_same dir">
### Mkdocs Simple <a name="mkdocs.plugins.mkdocs_simple">
### Mkdocs Material <a name="mkdocs.plugins.mkdocs_material">
### Mkdocstrings <a name="mkdocs.plugins.mkdocstrings">
### Mkdocstrings python legacy <a name="mkdocs.plugins.mkdocstrings_python_legacy">
### Pytkdocs <a name="mkdocs.plugins.pytkdocs">
## Resources <a name="mkdocs.resources">
