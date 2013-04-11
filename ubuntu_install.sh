# AMI: ami-3fec7956

sudo apt-get update
sudo apt-get install git python-numpy python-scipy python-matplotlib python-h5py redis-server python-setuptools python-pip
sudo pip install virtualenv

virtualenv --system-site-packages padeenv

git clone 

wget https://github.com/itmat/pade/archive/master.tar.gz
tar zxvf master.tar.gz
cd pade-master
python setup.py install
