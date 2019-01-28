#define VIENNACL_WITH_UBLAS 1
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <vector>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp" 
#include "viennacl/linalg/prod.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/ell_matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/matrix_operations.hpp"
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "viennacl/tools/timer.hpp"
#include "viennacl/io/matrix_market.hpp"
using namespace std;
using namespace boost::numeric::ublas;


const unsigned int dimrow=256;
const unsigned int dimcol=800;
const unsigned int dimin=500;
const double sparsity=.5;

//matrix print function
void print_dense_matrix(viennacl::matrix_base<double> mat,int row, int col){
 cout<<"=========="<<endl;
 for(int i=0;i<row;i++){
  for(int j=0;j<col;j++){
   cout<<mat(i,j)<<"\t";
  }
  cout<<endl;
 }
 cout<<"=========="<<endl;
}

#define BENCHMARK_RUNS          250



inline void printOps(double num_ops, double exec_time)
{
 std::cout << "GFLOPs: " << num_ops / (1000000 * exec_time * 1000) << std::endl;
}

template<typename ScalarType>
int run_benchmark() {
 viennacl::tools::timer timer;
 double exec_time;

 viennacl::matrix_base<ScalarType> in_matrix(dimrow,dimin,true);
 viennacl::matrix_base<ScalarType> out_matrix(dimrow,dimcol,true);
 viennacl::compressed_matrix<ScalarType> vcl_compressed_matrix(dimin,dimcol);

// if (!viennacl::io::read_matrix_market_file(in_matrix, "../examples/testdata/mat65k.mtx")) {
//  std::cout << "Error reading Matrix file" << std::endl;
//  return 0;
// }
// //unsigned int cg_mat_size = cg_mat.size();
// std::cout << "done reading matrix" << std::endl;

 srand(time(NULL));

 for(int i=0; i<dimrow; ++i) {
     for(int j=0; j<dimin; ++j) {
         if(rand()/RAND_MAX < sparsity) {
             in_matrix(i,j) = 0;
         }
         else {
             in_matrix(i, j) = rand() / RAND_MAX - 0.5;
             //vcl_compressed_matrix(i,j) = in_matrix(i, j);
         }
     }
 }

    for(int i=0; i<dimin; ++i) {
        for(int j=0; j<dimcol; ++j) {
            if(rand()/RAND_MAX < sparsity) {
                1;
            }
            else {
                //in_matrix(i, j) = rand() / RAND_MAX - 0.5;
                vcl_compressed_matrix(i,j) = rand() / RAND_MAX - 0.5;
            }
        }
    }



 //cpu to gpu:
// viennacl::copy(in_matrix, vcl_compressed_matrix);



 ///////////// Matrix operations /////////////////

 std::cout << "------- Matrix-Vector product with compressed_matrix ----------" << std::endl;


 viennacl::backend::finish();
 timer.start();
 for (int runs = 0; runs < BENCHMARK_RUNS; ++runs) {
  viennacl::linalg::prod_impl(in_matrix, vcl_compressed_matrix, false, out_matrix);
 }
 viennacl::backend::finish();
 exec_time = timer.get();

 std::cout << "GPU time D*S " << exec_time << std::endl;
 std::cout <<  std::endl << std::endl;

 viennacl::backend::finish();
 timer.start();
 for (int runs = 0; runs < BENCHMARK_RUNS; ++runs) {
  viennacl::linalg::prod_impl(in_matrix, vcl_compressed_matrix, true, out_matrix);
 }
 viennacl::backend::finish();
 exec_time = timer.get();

 std::cout << "GPU time D*trans(S) " << exec_time << std::endl;
 std::cout <<  std::endl << std::endl;

// std::cout << "GPU align1 ";
// printOps(2.0 * static_cast<double>(in_matrix.nnz()),
//          static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
// std::cout << vcl_vec1[0] << std::endl;


}




int main (){

 run_benchmark<double>();

     std::cout << viennacl::ocl::current_device().info() << std::endl;

     matrix<double> cpu_matrix (2,2);


     //initializing Dense GPU matrix
     viennacl::matrix_base<double>  matD(2,2,true);
     matD(0,0) = 1.;
     matD(0,1) = 2.;
     matD(1,0) = 3.;
     matD(1,1) = 4.;
     std::cout << "matD " << matD << endl;

     //initializing Sparse GPU matrix(by copy)
     viennacl::compressed_matrix<double>  matS(2,2);
     cpu_matrix(0,0) = -5.;
     cpu_matrix(0,1) = 7.;
     cpu_matrix(1,0) = 1.;
     cpu_matrix(1,1) = 0.;
     viennacl::copy(cpu_matrix, matS);
     std::cout << "matS " << matS << endl;

     //initializing Dense result GPU matrix
     viennacl::matrix_base<double>  matDr(2,2,true);


     //S*D
     //viennacl::linalg::prod_impl(matS, matD, matDr);
     //viennacl::linalg::prod_impl(matD, matS, false, matDr);
     viennacl::matrix_base<double> matD_tmp(2,2,true);
     viennacl::copy_impl(matS, matD_tmp);
     //matDr = matD * matD_tmp; //viennacl::linalg::prod_impl(matD, matD_tmp, matDr);
     //print_dense_matrix(matDr, 2, 2);
     std::cout << "matS" << matS << endl;
    std::cout << "matD_tmp" << matD_tmp << endl;


     //S*tr(D)
     //viennacl::linalg::prod_impl(matS, trans(matD), matDr);
     //viennacl::linalg::prod_impl(matD, matS, true, matDr);
     //print_dense_matrix(matDr, 2, 2);
     viennacl::ocl::get_queue().finish();
     //std::cout << "result " << matDr << endl;



 
}
