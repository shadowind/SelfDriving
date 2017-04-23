# Terminal multiplexer

[image]
tmux is a "terminal multiplexer", it enables a number of terminals (or windows)
to be accessed and controlled from a single terminal. tmux is intended to be a
simple, modern, BSD-licensed alternative to programs such as GNU screen.

This release runs on OpenBSD, FreeBSD, NetBSD, Linux, OS X and Solaris.


Installation guide
https://gist.github.com/simme/1297707

I faced a problem for accessing brew/homebrew folder. 
```
$ cd /usr/local/library/Homebrew -- depending on where you installed the homebrew  
$ git reset --hard  
$ git clean -df
$ brew update
```

Instruction for pressing keys
http://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/ % create a horizontal pane  
All commands in tmux are triggered by a prefix key followed by a command key (quite similar to emacs). By default, tmux uses C-b as prefix key. This notation might read a little weird if you’re not used to it. In this emacs notation C- means “press and hold the Ctrl key”3. Thus C-b simply means press the Ctrl and b keys at the same time.

```
" create a vertical pane
h move to the left pane. *  
j move to the pane below *  
l move to the right pane *
k move to the pane above *
k move to the pane above *
q show pane numbers
o toggle between panes
} swap with next pane
{ swap with previous pane
```

