apt-add-repository ppa:deadsnakes/ppa
apt-get update
apt-get install python3.9 python3.9-dev python3.9-distutils pandoc --yes
# Create a symbolic link so "python" command points to python 3.9
ln -T /usr/bin/python3.9 /usr/bin/python
# Install Jupyter Lab
python -m pip install jupyterlab