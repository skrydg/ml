# git
sudo yum -y install git
git clone https://skrydg:JNTyTLaEMV496Re@github.com/skrydg/ml.git
cd ml
git config --global user.email "skrrydg@yandex.ru"
git config --global user.name "skrrydg"

# ifconfig
sudo yum -y install net-tools
echo -e "export PATH=\$PATH:/sbin\n" >> ~/.bashrc

# timezone settings
sudo yum install dbus
sudo timedatectl set-timezone Europe/Moscow

# pip
sudo yum install python3 -y
echo -e "alias python=python3\n" >> ~/.bashrc
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py

# python libs
pip install tornado
pip install requests
pip3 install numpy
pip3 install matplotlib
pip3 install sklearn
pip3 install pandas
pip3 install ortools
pip3 install pulp

# install jupiter notebook
pip install jupyterlab

# install psutil
sudo yum install python-dev -y
sudo yum install gcc -y
sudo yum -y install python3 python-dev python3-dev \
     build-essential libssl-dev libffi-dev \
     libxml2-dev libxslt1-dev zlib1g-dev \
     python-pip
pip3 install psutil