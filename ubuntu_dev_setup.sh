# AMI: ami-3fec7956
set -e

# Install system-wide packages
sudo apt-get update
sudo apt-get install git python-numpy python-scipy python-matplotlib python-h5py redis-server python-setuptools python-pip

# Install virtualenv, set up an environment, and activate it
sudo pip install virtualenv
virtualenv --system-site-packages padeenv
source padeenv/bin/activate

# Clone the repo
git clone https://github.com/itmat/pade.git

cd pade
python setup.py develop
