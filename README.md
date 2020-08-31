# prod-computer-vision
Facial Detection, Machine Learning

## Development Environment

pythond on MAC

use the brew to manage the python versioning
```
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

install other requirements
```
pip install opencv-python
pip install matplotlib pandas numpy pillow scipy torch torchvision
```
