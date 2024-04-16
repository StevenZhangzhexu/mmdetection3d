conda create --name openmmlab python=3.8 -y
conda activate openmmlab

pip install torch==1.13.1+cu117 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4'
mim install 'mmdet>=3.0.0'

pip install laspy laszip

pip install -v -e .


############### optional ##################
#
## mim install "mmdet3d>=1.1.0"
#
#pip install cumm-cu117
#pip install spconv-cu117
#
#conda install openblas-devel -c anaconda
#export CPLUS_INCLUDE_PATH=CPLUS_INCLUDE_PATH:/home/pc1/miniconda3/envs/openmmlab/include
## replace ${YOUR_CONDA_ENVS_DIR} to your anaconda environment path e.g. `/home/username/anaconda3/envs/openmmlab`.
#export CUDA_HOME=/usr/local/cuda-11.7
#sudo apt install build-essential python3-dev libopenblas-dev
#pip install ninja
#pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps
#
#sudo apt-get install libsparsehash-dev
#pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
#
## conda install -c conda-forge mmcv-full
