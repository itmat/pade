## Install PADE on your mac

1. Make sure you have python v2.7.5 or higher.
    
    brew install python

2. Install pip

    sudo easy_install pip

3. Install and setup a virtualenv 

    sudo easy_install virtualenv
    virtualenv /Users/<username>/my_python-2.7.5 --system-site-packages
    source /Users/<username>/my_python-2.7.5/bin/activate

4. Install required dependencies

    brew install gcc
    pip install numpy 
    pip install scipy
    pip install pyyaml
    pip install setuptools
    brew tap homebrew/science
    brew install hdf5
    pip install h5py
    pip install jinja2
    pip install matplotlib

5. Git clone and install PADE

    git clone git@github.com:itmat/pade.git
    cd pade/
    python setup.py install
