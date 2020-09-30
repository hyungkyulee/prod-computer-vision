# prod-computer-vision
Facial Detection, Machine Learning

## Development Environment

python and pip on MAC
```
# use the brew to manage the python versioning
$ brew list | grep python
python

$ brew info python
python: stable 3.7.3 (bottled), HEAD
Interpreted, interactive, object-oriented programming language
https://www.python.org/
/usr/local/Cellar/python/3.7.2_1 (8,437 files, 118MB) *
## further output not included ##

$ brew update && brew upgrade python

# If you added the previous alias, use a text editor to update the line to the following
alias python=/usr/local/bin/python3

# Note that this is a capital V (not lowercase)
$ pip -V
pip 19.0.3 from /Library/Python/2.7/site-packages/pip-19.0.3-py2.7.egg/pip (python 2.7)

# To ensure we're installing packages compatible with our new version of Python
$ which pip3
/usr/local/bin/pip3

# add it to our shell configuration file
$ echo "alias pip=/usr/local/bin/pip3" >> ~/.zshrc 
# or for Bash
$ echo "alias pip=/usr/local/bin/pip3" >> ~/.bashrc

# confirm that running pip points to pip3 by opening a new shell or by resetting our current shell
# This command reloads the current shell without exiting the session
# Alternatively, exit the shell and start a new one
$ exec $0
$ chsh -s /bin/zsh

# Now we can look to see where pip points us
$ which pip
pip: aliased to /usr/local/bin/pip3
```
 (* don't do manually move the symlink or macOS python 3 installer: https://opensource.com/article/19/5/python-3-default-mac)
 
install anaconda
```
brew cask install anaconda
export PATH="/usr/local/anaconda3/bin:$PATH"
```
 (*reference : https://docs.anaconda.com/anaconda/install/mac-os/)
 
install PyTorch and torchvision
```
conda install pytorch torchvision -c pytorch
```

install cuda (* ref: https://github.com/pytorch/pytorch#from-source)
```
# Add these packages if torch.distributed is needed
conda install pkg-config libuv
```

install other requirements
```
pip install opencv-python
pip install matplotlib pandas numpy pillow scipy torch torchvision
pip install torchsummary
```
