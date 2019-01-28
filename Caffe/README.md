# Lite version of OpenCL Caffe

support only GPU mode

## Dependency
- customized viennaCL
- OpenCL

## Build

```
cmake -DUSE_INDEX_64=OFF -DUSE_CUDA=OFF -DUSE_GREENTEA=ON -DCPU_ONLY=OFF -DUSE_OPENCV=ON -DBUILD_docs=OFF -DBUILD_python=OFF -DUSE_CLBLAST=OFF -DDISABLE_DEVICE_HOST_UNIFIED_MEMORY=ON -DViennaCL_INCLUDE_DIR=/example/viennacl-dev -DOPENCL_LIBRARIES=/example/libOpenCL.so -DOPENCL_INCLUDE_DIRS=/example/viennacl-dev ..
```

## Regularization Type: L1 Proximal Operator

### Usage
On the `solver.prototxt`, add below parameter.
```prototype
regularization_type : "ProxL1"
proximal_decay : 1.5
```



