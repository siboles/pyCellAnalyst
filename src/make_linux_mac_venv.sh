#!/bin/bash

pip install virtualenv virtualenvwrapper

echo "export WORKON_HOME=$HOME/Envs" >> $HOME/.profile
echo "source /usr/local/bin/virtualenvwrapper.sh" >> $HOME/.profile

source $HOME/.profile
