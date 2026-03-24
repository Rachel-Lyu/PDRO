# PDRO
conda create -n pdro python=3.10 -y
conda activate pdro

git clone https://github.com/zykls/folktables.git
cd folktables
pip install .

conda install numpy pandas matplotlib scikit-learn -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y