FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Install base apt packages
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libopenblas-dev

# Install MinkowskiEngine
RUN TORCH_CUDA_ARCH_LIST="6.1 7.0 8.6 9.0" \
    pip install git+https://github.com/daizhirui/MinkowskiEngine.git@ce930eeb403a8e3f99693662ec5ce329a0ab3528 -v --no-deps \
    --global-option="--blas=openblas" \
    --global-option="--force_cuda"

# Install OpenMMLab projects
RUN pip install --no-deps \
    mmengine==0.9.0 \
    mmdet==3.3.0 \
    mmsegmentation==1.2.0 \
    mmdet3d==1.4.0 \
    mmpretrain==1.2.0

# Install mmcv
RUN git clone https://github.com/open-mmlab/mmcv.git \
    && cd mmcv \
    && git reset --hard 780ffed9f3736fedadf18b51266ecbf521e64cf6 \
    && sed -i "s/'-std=c++14'] if cuda_args else/'-std=c++14', '-arch=sm_90'] if cuda_args else/g" setup.py \
    && TORCH_CUDA_ARCH_LIST="6.1 7.0 8.6 9.0" \
    && pip install -v -e . --no-deps \
    && cd ..

# Install torch-scatter 
RUN pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html --no-deps

# Install ScanNet superpoint segmentator
RUN git clone https://github.com/Karbo123/segmentator.git \
    && cd segmentator/csrc \
    && git reset --hard 76efe46d03dd27afa78df972b17d07f2c6cfb696 \
    && sed -i "s/set(CMAKE_CXX_STANDARD 14)/set(CMAKE_CXX_STANDARD 17)/g" CMakeLists.txt \
    && mkdir build \
    && cd build \
    && cmake .. \
        -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
        -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
        -DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'` \
    && make \
    && make install \
    && cd ../../..

# Install remaining python packages
RUN pip install --no-deps \
    spconv-cu120==2.3.6 \
    addict==2.4.0 \
    yapf==0.33.0 \
    termcolor==2.3.0 \
    packaging==23.1 \
    numpy==1.24.1 \
    rich==13.3.5 \
    opencv-python==4.7.0.72 \
    pycocotools==2.0.6 \
    Shapely==1.8.5 \
    scipy==1.10.1 \
    terminaltables==3.1.10 \
    numba==0.57.0 \
    llvmlite==0.40.0 \
    pccm==0.4.7 \
    ccimport==0.4.2 \
    pybind11==2.10.4 \
    ninja==1.11.1 \
    lark==1.1.5 \
    cumm-cu120==0.5.1 \
    pyquaternion==0.9.9 \
    lyft-dataset-sdk==0.0.8 \
    pandas==2.0.1 \
    python-dateutil==2.8.2 \
    matplotlib==3.5.2 \
    pyparsing==3.0.9 \
    cycler==0.11.0 \
    kiwisolver==1.4.4 \
    scikit-learn==1.2.2 \
    joblib==1.2.0 \
    threadpoolctl==3.1.0 \
    cachetools==5.3.0 \
    nuscenes-devkit==1.1.10 \
    trimesh==3.21.6 \
    open3d==0.17.0 \
    plotly==5.18.0 \
    dash==2.14.2 \
    plyfile==1.0.2 \
    flask==3.0.0 \
    werkzeug==3.0.1 \
    click==8.1.7 \
    blinker==1.7.0 \
    itsdangerous==2.1.2 \
    importlib_metadata==2.1.2 \
    zipp==3.17.0 \
    natsort==8.4.0 \
    timm==0.9.16 \
    imageio==2.34.0 \
    portalocker==2.8.2 \
    ftfy==6.2.0 \
    regex==2024.4.16