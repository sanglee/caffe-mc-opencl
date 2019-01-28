# OpenCL Caffe

support GPU mode only



## Build

```
cmake -DUSE_INDEX_64=OFF -DUSE_CUDA=OFF -DUSE_GREENTEA=ON -DCPU_ONLY=OFF -DUSE_OPENCV=ON -DBUILD_docs=OFF -DBUILD_python=OFF -DUSE_CLBLAST=OFF -DDISABLE_DEVICE_HOST_UNIFIED_MEMORY=ON -DViennaCL_INCLUDE_DIR=/[yourpath]/viennacl-dev -DOPENCL_LIBRARIES=/[yourpath]/libOpenCL.so -DOPENCL_INCLUDE_DIRS=/[yourpath]/viennacl-dev ..
```

## Regularization Type: L1 Proximal Operator

```prototype
regularization_type : "ProxL1"
proximal_decay : 1.5
```



