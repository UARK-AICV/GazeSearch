# Installation

```bash
conda env remove --name chestsearch
conda create -n chestsearch python=3.8 -y
conda activate chestsearch
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -y

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
cd ./chestsearch/pixel_decoder/ops
sh make.sh


pip install timm
pip install scipy
pip install opencv-python
pip install wget
pip install setuptools==59.5.0
pip install einops
pip install protobuf==4.25.0
```