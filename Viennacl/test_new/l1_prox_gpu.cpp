#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <vector>
#include <iostream>
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
//#define VIENNACL_WITH_UBLAS 1
#define VIENNACL_WITH_OPENCL 1
//#include <boost/numeric/ublas/triangular.hpp>
//#include <boost/numeric/ublas/vector.hpp>
//#include <boost/numeric/ublas/vector_proxy.hpp>
//#include <boost/numeric/ublas/matrix_sparse.hpp>
//#include <boost/numeric/ublas/operation_sparse.hpp>
//#include <boost/numeric/ublas/lu.hpp>
#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
using namespace std;
using namespace boost::numeric::ublas;

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


int main (){
 matrix<double> cpu_matrix (2,2);


 //initializing Dense GPU matrix
 viennacl::matrix_base<double>  matD(2,2,true);
 matD(0,0) = 1.;
 matD(0,1) = -1.;
 matD(1,0) = 1.;
 matD(1,1) = 1.;


 //initializing Sparse GPU matrix(by copy)
 viennacl::compressed_matrix<double>  matS(2,2);
 cpu_matrix(0,0) = -1.;
 cpu_matrix(0,1) = 0.;
 cpu_matrix(1,0) = 1.;
 cpu_matrix(1,1) = 0.;
 viennacl::copy(cpu_matrix, matS);


 //initializing Dense result GPU matrix
 viennacl::matrix_base<double>  matDr(2,2,true);


 //S*D
 viennacl::linalg::prod_impl(matS, matD, matDr);
 print_dense_matrix(matDr, 2, 2);


 //S*tr(D)
 viennacl::linalg::prod_impl(matS, trans(matD), matDr);
 print_dense_matrix(matDr, 2, 2);

 viennacl::matrix_expression<viennacl::matrix_base<double>,viennacl::matrix_base<double>,viennacl::op_acos> matE(matD,matD);
// matD = matE;
// cout<<viennacl::op_acos<<endl;
 print_dense_matrix(matD, 2, 2);
viennacl::linalg::element_op(matD,viennacl::matrix_expression<const viennacl::matrix_base<double>,const viennacl::matrix_base<double>,viennacl::op_element_unary<viennacl::op_acos>>(matD,matD));

 print_dense_matrix(matD, 2, 2);
}
