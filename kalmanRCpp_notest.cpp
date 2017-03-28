/*
Efficient Kalman - Linear and Extended Kalman Filter
Haonan Zhou
v0.1.1, 03/28/2017

This code interacts with R interface and implements Kalman Filter and Smoother 
through the usage of Rcpp and RcppArmadillo. 

To do's:
- Compare with R benchmark function to demonstrate its efficiency;
- Complete documentation;
- Possible extension into package / MLE/EM estimation, etc.
- Further debugging. 
*/

#include <iostream>
#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
List kalmanC(const arma::colvec& x0, const arma::mat& y, 
                 const arma::mat& Sigma0, const arma::mat& Phi, 
                 const arma::mat& A, const arma::mat& Q, const arma::mat& R, 
                 bool smooth){
  
  // Declare variables to use
  arma::mat P = Sigma0;
  arma::colvec x = x0;
  
  arma::uword xn = x0.n_rows;   // Number of states
  arma::colvec xpred_temp(xn);
  
  arma::mat Ppred(xn,xn);
  arma::uword k = y.n_rows;     // Number of Periods
  arma::mat K(xn,k);
  arma::mat xmat(k,xn);          // Store filtered series
  arma::mat xpred(k,xn);         // Store predicted series
  
  arma::colvec eps(k);
  arma::mat Sigma(k,k);
  
  int err = 0;
  double ll = 0;
  
  arma::cube Ppredfield(xn,xn,k);
  arma::cube Pmatfield(xn,xn,k);
  
  // Filter (forward)
  for (arma::uword i = 0; i < k; ++i){
    err = 0;
    // Forecasting step
    xpred_temp = Phi * x;
    xpred.row(i) = xpred_temp.t();
    Ppred = Phi * P * Phi.t() + Q;
    
    // Update step
    Sigma = A * Ppred * A.t() + R;
    try {
      K = Ppred * A.t() * arma::inv(Sigma);
    } catch (const std::exception &exc){
      K = Ppred * A.t() * arma::pinv(Sigma);
      Rcout << "Singular state covariance matrix detected, use generalized inverse instead." << std::endl;
      int err = 1;
    }
    
    eps = y.row(i) - A * xpred_temp;
    x = xpred_temp + K * eps; 
    P = (arma::eye<arma::mat>(xn,xn) - K * A) * Ppred;
    
    xmat.row(i) = x.t();
    
    // Update likelihood function
    double val;
    double sign;
    arma::log_det(val, sign, Sigma);
    if (err == 0){
      ll += arma::as_scalar(eps.t() * arma::inv(Sigma) * eps) + val*sign;
    } else{
      ll += arma::as_scalar(eps.t() * arma::pinv(Sigma) * eps) + val*sign;
    }
    
    // Store all objects into corresponding fields
    Pmatfield.slice(i) = P;
    Ppredfield.slice(i) = Ppred;
  }
  
  // Smooth (backward)
  if (!smooth){
    // Smoothing not required. Return NULL in corresponding field
    return List::create(Named("xfilt") = xmat,
                        Named("xpred") = xpred,
                        Named("xsmooth") = R_NilValue,
                        Named("ll") = ll, 
                        Named("P") = wrap(Pmatfield),
                        Named("Ppred") = wrap(Ppredfield),
                        Named("Psmooth") = R_NilValue);
  } else {
    arma::mat xsmooth(k, xn);
    arma::mat Psmooth(xn,xn);
    arma::mat J(xn, xn);      // J matrix in Shumway and Stoffer (2013)
    arma::colvec xsmooth_temp(xn);
    
    arma::cube Psmoothfield(xn, xn, k);
    Psmoothfield.slice(k-1) = Pmatfield.slice(k-1);
    xsmooth.row(k-1) = xmat.row(k-1);
    
    for (arma::uword j = k-1; j > 0; j--){
      try {
        J = Pmatfield.slice(j-1) * Phi.t() * arma::inv(Ppredfield.slice(j));
      } catch (const std::exception &exc){
        J = Pmatfield.slice(j-1) * Phi.t() * arma::pinv(Ppredfield.slice(j));
        Rcout << "Singular matrix detected, use generalized inverse instead." << std::endl;
      }
      xsmooth_temp = xmat.row(j-1).t() + J * (xsmooth.row(j).t() - xpred.row(j).t());
      xsmooth.row(j-1) = xsmooth_temp.t();
      Psmooth = Pmatfield.slice(j-1) + J * (Psmoothfield.slice(j) - Ppredfield.slice(j)) * J.t();
      
      Psmoothfield.slice(j-1) = Psmooth;
    }
    
    return List::create(Named("xfilt") = xmat,
                        Named("xpred") = xpred,
                        Named("xsmooth") = xsmooth,
                        Named("ll") = ll,
                        Named("P") = wrap(Pmatfield),
                        Named("Ppred") = wrap(Ppredfield),
                        Named("Psmooth") = wrap(Psmoothfield));
  }
  
}


// For our extended Kalman, each step needs to call R function to make prediction and 
// perform Jacobian calculation.

// [[Rcpp::export]]
List extkalmanC(const arma::colvec& x0, const arma::mat& y, 
             const arma::mat& Sigma0, const arma::mat& Q, const arma::mat& R, 
             bool smooth, Function f, Function h){
  Environment numDeriv("package:numDeriv");
  Function jacobian = numDeriv["jacobian"];
  
  // Declare variables to use
  arma::mat P = Sigma0;
  arma::colvec x = x0;
  
  arma::uword xn = x0.n_rows;   // Number of states
  arma::colvec xpred_temp(xn);
  
  arma::mat Ppred(xn,xn);
  arma::uword k = y.n_rows;     // Number of Periods
  arma::mat K(xn,k);
  arma::mat xmat(k,xn);          // Store filtered series
  arma::mat xpred(k,xn);         // Store predicted series
  
  arma::colvec eps(k);
  arma::mat Sigma(k,k);
  
  double ll = 0;
  int err = 0;
  
  arma::cube Ppredfield(xn,xn,k);
  arma::cube Pmatfield(xn,xn,k);
  
  arma::cube Afield(y.n_cols,xn,k);
  arma::cube Phifield(xn,xn,k);
  
  arma::mat Phi(xn, xn);
  arma::mat A(y.n_cols,xn);
  
  // Filter (forward)
  for (arma::uword i = 0; i < k; ++i){
    err = 0;
    // Forecasting step
    
    // For extended filter: evaluate function f and calculate Jacobian
    xpred_temp = as<arma::colvec>(f(x));
    xpred.row(i) = xpred_temp.t();
    
    Phi = as<arma::mat>(jacobian(_["func"] = f, _["x"] = x));
    Ppred = Phi * P * Phi.t() + Q;
    
    // Update step
    A = as<arma::mat>(jacobian(_["func"] = h, _["x"] = xpred_temp));
    
    Sigma = A * Ppred * A.t() + R;
    
    try {
      K = Ppred * A.t() * arma::inv(Sigma);
    } catch (const std::exception &exc){
      K = Ppred * A.t() * arma::pinv(Sigma);
      Rcout << "Singular state covariance matrix detected, use generalized inverse instead." << std::endl;
      err = 1;
    }
    
    eps = y.row(i) - A * xpred_temp;
    x = xpred_temp + K * eps; 
    P = (arma::eye<arma::mat>(xn,xn) - K * A) * Ppred;
    
    xmat.row(i) = x.t();
    
    // Update likelihood function
    double val;
    double sign;
    arma::log_det(val, sign, Sigma);
    if (err == 0){
      ll += arma::as_scalar(eps.t() * arma::inv(Sigma) * eps) + val*sign;
    } else{
      ll += arma::as_scalar(eps.t() * arma::pinv(Sigma) * eps) + val*sign;
    }
    
    // Store all objects into corresponding fields
    Pmatfield.slice(i) = P;
    Ppredfield.slice(i) = Ppred;
    Afield.slice(i) = A;
    Phifield.slice(i) = Phi;
  }
  
  // Smooth (backward)
  if (!smooth){
    // Smoothing not required. Return NULL in corresponding field
    return List::create(Named("xfilt") = xmat,
                        Named("xpred") = xpred,
                        Named("xsmooth") = R_NilValue,
                        Named("ll") = ll, 
                        Named("P") = wrap(Pmatfield),
                        Named("Ppred") = wrap(Ppredfield),
                        Named("Psmooth") = R_NilValue,
                        Named("Phifilt") = wrap(Phifield),
                        Named("Afilt") = wrap(Afield),
                        Named("Phismooth") = R_NilValue);
  } else {
    arma::mat xsmooth(k, xn);
    arma::mat Psmooth(xn,xn);
    arma::mat J(xn, xn);      // J matrix in Shumway and Stoffer (2013)
    arma::colvec xsmooth_temp(xn);
    
    arma::cube Psmoothfield(xn, xn, k);
    Psmoothfield.slice(k-1) = Pmatfield.slice(k-1);
    xsmooth.row(k-1) = xmat.row(k-1);
    
    arma::cube Phismoothfield(xn, xn, k);
    Phismoothfield.slice(k-1) = Phifield.slice(k-1);
    
    arma::mat Phismooth(xn, xn);
    
    for (arma::uword j = k-1; j > 0; j--){
      // Evaluate function
      Phismooth = Phifield.slice(j);
      try {
        J = Pmatfield.slice(j-1) * Phismooth.t() * arma::inv(Ppredfield.slice(j));
      } catch (const std::exception &exc){
        J = Pmatfield.slice(j-1) * Phismooth.t() * arma::pinv(Ppredfield.slice(j));
        Rcout << "Singular matrix detected, use generalized inverse instead." << std::endl;
      }

      xsmooth_temp = xmat.row(j-1).t() + J * (xsmooth.row(j).t() - xpred.row(j).t());
      xsmooth.row(j-1) = xsmooth_temp.t();
      Psmooth = Pmatfield(j-1) + J * (Psmoothfield.slice(j) - Ppredfield.slice(j)) * J.t();
      
      Psmoothfield.slice(j-1) = Psmooth;
      Phismoothfield.slice(j-1) = Phismooth;
    }
    
    return List::create(Named("xfilt") = xmat,
                        Named("xpred") = xpred,
                        Named("xsmooth") = xsmooth,
                        Named("ll") = ll,
                        Named("P") = wrap(Pmatfield),
                        Named("Ppred") = wrap(Ppredfield),
                        Named("Psmooth") = wrap(Psmoothfield),
                        Named("Phifilt") = wrap(Phifield),
                        Named("Afilt") = wrap(Afield),
                        Named("Phismooth") = wrap(Phismoothfield));
  }
}
