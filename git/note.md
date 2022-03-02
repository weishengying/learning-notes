# 杂记
1. 拉取别人的pr   git fetch remote pull/$pr_id/head:$new_name
2. 删除远程分支: git push origin --delete [branchname]

# git stash 详解
[参考](https://blog.csdn.net/stone_yw/article/details/80795669)
## 应用场景
1 当正在dev分支上开发某个项目，这时项目中出现一个bug，需要紧急修复，但是正在开发的内容只是完成一半，还不想提交，这时可以用git stash命令将修改的内容保存至堆栈区，然后顺利切换到hotfix分支进行bug修复，修复完成后，再次切回到dev分支，从堆栈中恢复刚刚保存的内容。
2 由于疏忽，本应该在dev分支开发的内容，却在master上进行了开发，需要重新切回到dev分支上进行开发，可以用git stash将内容保存至堆栈中，切回到dev分支后，再次恢复内容即可。
总的来说，git stash命令的作用就是将目前还不想提交的但是已经修改的内容进行保存至堆栈中，后续可以在某个分支上恢复出堆栈中的内容。这也就是说，stash中的内容不仅仅可以恢复到原先开发的分支，也可以恢复到其他任意指定的分支上。git stash作用的范围包括工作区和暂存区中的内容，也就是说没有提交的内容都会保存至堆栈中。

## 命令详解
### 1 git stash
能够将所有未提交的修改（工作区和暂存区）保存至堆栈中，用于后续恢复当前工作目录。

### 2 git stash save
作用等同于git stash，区别是可以加一些注释，如：git stash save “test1”

### 3 git stash list
查看当前stash中的内容

### 4 git stash pop
将当前stash中的内容弹出，并应用到当前分支对应的工作目录上。
注：该命令将堆栈中最近保存的内容删除（栈是先进后出）

### 5 git stash apply
将堆栈中的内容应用到当前目录，不同于git stash pop，该命令不会将内容从堆栈中删除，也就说该命令能够将堆栈的内容多次应用到工作目录中，适应于多个分支的情况。
堆栈中的内容并没有删除。
可以使用git stash apply + stash名字（如stash@{1}）指定恢复哪个stash到当前的工作目录。

### 6 git stash drop + 名称
从堆栈中移除某个指定的stash

### 7 git stash clear
清除堆栈中的所有 内容

### 8 git stash show
查看堆栈中最新保存的stash和当前目录的差异。
git stash show stash@{1}查看指定的stash和当前目录差异。
通过 git stash show -p 查看详细的不同，
同样，通过git stash show stash@{1} -p查看指定的stash的差异内容。

### 9 git stash branch
从最新的stash创建分支。
应用场景：当储藏了部分工作，暂时不去理会，继续在当前分支进行开发，后续想将stash中的内容恢复到当前工作目录时，如果是针对同一个文件的修改（即便不是同行数据），那么可能会发生冲突，恢复失败，这里通过创建新的分支来解决。可以用于解决stash中的内容和当前目录的内容发生冲突的情景。
发生冲突时，需手动解决冲突。
