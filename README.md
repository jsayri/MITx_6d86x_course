# MITx 6.86x course. Projects & lectures code repository
Online course of Machine Learning with Python, From Linear Models to Deep Learning, by MIT with the EDX platform.
Repository maintanied by [Jose](https://github.com/jsayri), [Ravi](https://github.com/vasurav) and [Sophia](https://github.com/sophiafakih) 

## Global objective for this repository
Over the next months some python's code and jupyter implementation will be developed. In order to track the evolution of the participants to this repository, a common folder tree directory was proposed. The general idea is to share our codes and solutions while having a backup during the course development.

## What to store & folders architecture
Is expected to store the __code solution__ or any implementation done during the assitance to the lectures. In order to have a common workframe we propose to follow this architecture:
```
Main folder
│   Readme.md
│   environment.yaml
│   .gitignore
│   ...
└─── units
    └─── unit_1
        │ python code from lectures
        │ python code for testing
        │ ...
        └─── proyect
            │ python code for project
            │ results from project (text files, no big images)
            │ ...
```

__Avoid to store dataset and large files__ (pdf, ppt, doc, etc). Better to keep track of files that can be visulized directly from the github server.

## To prepare your local repository
In order to set-up your local & remote space, some steps are recomended:
1. __Each participant should create a branch based on the last version of the master.__
    Create a local branch by doing:
    ```git
    git checkout -b NEW-BRANCH-NAME     # create and switch to that branch
    ```
    or by doing:
    ```git
    git branch NEW-BRANCH-NAME          # create a local branch
    git checkout NEW-BRANCH-NAME        # switch to "NEW-BRANCH-NAME" branch
    ```
    Example:
    supose that a new branch is creater under the name of test_branch:
    ```git
    git checkout -b test_branch
    ```
    
2. __Organize your directory, commit your results and export your local branch to the github server.__
    ```git
    git push --set-upstream <remote-name> <local-branch-name> 
    ```
    Example continuation:
    ```git
    git push --set-upstream origin test_branch 
    ```
3. __After branch set-up, commons commit and push are:__
    To track and update our work on a define file, we can update the git track index as:
    ``` git
    git commit <filename.extension> -m "description message for the update of a unique file"
    ```
    In the case of the modification of multiple files and addition of new ones, we could commit the group as:
    ```git
    git commit -a -m "description message for the update of a group of files and/or added files."
    ```
    To sent modification to the server of our current branch, we guarantee that we are at the working branch and wewrite:
    ``` git
    git push
    ```
    
## Required libraries, environment discussion
All libraries and require packages all listed inside requirements.yalm file.
Recover and install environement as:
```console
> conda env create -f environment.yaml --force
```
Do not forget to update environment.yaml file if new packages were installed:
```console
> conda env export -n ds_gp -f environment.yaml
```

License
----
MIT
