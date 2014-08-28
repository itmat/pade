### Install PADE (cluster)

##### 1. Create a virtualenv with python v2.7.5 or higher
    
    bsub -Is bash
    module add python-2.7.5
    virtualenv $HOME/my_python-2.7.5 --system-site-packages
    source $HOME/my_python-2.7.5/bin/activate

##### 2. Clone the repository, install required dependencies and install PADE

    git clone git@github.com:itmat/pade.git
    cd pade/
    pip install jinja2 pyyaml matplotlib h5py
    python setup.py install


### Install PADE (mac)

##### 1. Make sure you have python v2.7.5 or higher

    brew install python

##### 2. Install pip

    sudo easy_install pip

##### 3. Install and setup a virtualenv 

    sudo easy_install virtualenv

You can use any other name for the virtualenv instead of "my_python-2.7.5".

    virtualenv /Users/<username>/my_python-2.7.5 --system-site-packages
    source /Users/<username>/my_python-2.7.5/bin/activate

##### 4. Install pip

    sudo easy_install pip

##### 5. Install required dependencies

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

##### 6. Git clone and install PADE

    git clone git@github.com:itmat/pade.git
    cd pade/
    python setup.py install
