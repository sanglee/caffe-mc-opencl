#g++ -std=c++11 -w -I/home/sklee/viennaCL/viennacl-dev/ -L/home/sklee/viennaCL/viennacl-dev/build/libviennacl -L/usr/local/cuda/lib64 -o prox_test l1_prox_gpu.cpp
g++ -std=c++11 -w -DVIENNACL_WITH_OPENCL vienna_product.cpp -I/home/sklee/viennaCL/viennacl-dev/ -L/usr/local/cuda/lib64/ -lOpenCL -o mult_test 
