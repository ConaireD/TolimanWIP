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

