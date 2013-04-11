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

echo
echo ------------------------- PADE ------------------------------------
echo
echo I have installed a full Python environment in padeenv. It contains
echo PADE as well as all of its dependencies.  To run pade, please first
echo activate the environment by running:
echo
echo  source padeenv/bin/activate
echo
echo Then you can simply type 'pade' to run it. You can learn how to use
echo pade by running 'pade -h' or viewing our documentation at
echo http://itmat.github.io/pade/.
echo
