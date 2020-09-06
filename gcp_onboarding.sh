# git
sudo apt-get -y install git
git clone https://skrydg:JNTyTLaEMV496Re@github.com/skrydg/ml.git
cd ml
git config --global user.email "skrrydg@yandex.ru"
git config --global user.name "skrrydg"

# ifconfig
sudo apt-get -y install net-tools
echo -e "export PATH=\$PATH:/sbin\n" >> ~/.bashrc

# timezone settings
sudo apt-get install dbus
sudo timedatectl set-timezone Europe/Moscow

sudo apt-get install python3-distutils -y

# pip
sudo apt-get install python3 -y
echo -e "alias python=python3\n" >> ~/.bashrc
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py

# python libs
sudo pip install tornado
sudo pip install requests
sudo pip3 install numpy
sudo pip3 install matplotlib
sudo pip3 install sklearn
sudo pip3 install pandas
sudo pip3 install ortools
sudo pip3 install pulp

sudo pip3 install nlp
sudo pip install -U spacy
python3 -m spacy download en

# install jupiter notebook
pip install jupyterlab

# install psutil
sudo apt-get install python-dev -y
sudo apt-get install gcc -y
sudo apt-get -y install python3 python-dev python3-dev \
     build-essential libssl-dev libffi-dev \
     libxml2-dev libxslt1-dev zlib1g-dev \
     python-pip
pip3 install psutil