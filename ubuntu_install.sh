# AMI: ami-3fec7956
apt-get install git python-numpy python-scipy python-matplotlib python-h5py python-nose redis-server python-setuptools
wget https://github.com/itmat/pade/archive/master.tar.gz
tar zxvf master.tar.gz
cd pade-master
python setup.py install
