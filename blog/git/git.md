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

