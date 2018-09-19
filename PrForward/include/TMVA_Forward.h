#pragma once

#include <vector>
#include <cmath>
#include <string>
#include <iostream>

#include "PrForwardConstants.h"


namespace SciFi {

struct TMVA {
  float fMin_1[3][7];
  float fMax_1[3][7];

  float fWeightMatrix0to1[11][8];   // weight matrix from layer 0 to 1
  float fWeightMatrix1to2[10][11];   // weight matrix from layer 1 to 2
  float fWeightMatrix2to3[8][10];   // weight matrix from layer 2 to 3
  float fWeightMatrix3to4[1][8];   // weight matrix from layer 3 to 4

  inline void Transform_1( float iv[7]) const
  {
    const int cls = 2; //what are the other???
    const int nVar = 7;
    
    for (int ivar=0;ivar<nVar;ivar++) {
      const float offset = fMin_1[cls][ivar];
      const float scale  = 1.0/(fMax_1[cls][ivar]-fMin_1[cls][ivar]); //TODO speed this up. but then not easy to update :(
      iv[ivar] = (iv[ivar]-offset)*scale * 2. - 1.;
    }
  }
  
  inline float ActivationFnc(float x) const {
     // rectified linear unit
     return x*(x>0);
  }

  //TODO do we need exp here? can we live with pure x output?!
  inline float OutputActivationFnc(float x) const {
     // sigmoid
     return 1.0/(1.0+exp(-x));
  }
  
  inline float GetMvaValue__( const float inputValues[7] ) const
  {
    //TODO check auto vectorization here: 'not vectorized: unsupported use in stmt' , 'Unsupported pattern'

     float fWeights0[8]  = {};
     float fWeights1[11] = {};
     float fWeights2[10] = {};
     float fWeights3[8]  = {};
     float fWeights4[1]  = {};

     fWeights0[7]  = 1.f;
     fWeights1[10] = 1.f;
     fWeights2[9]  = 1.f;
     fWeights3[7]  = 1.f;
     
     /// no offset node in output layer! i.e. no 1.f initialisation for fWeights4

     for (int i=0; i<7; i++)
        fWeights0[i]=inputValues[i];

     // layer 0 to 1
     for (int o=0; o<10; o++) {
        for (int i=0; i<8; i++) {
           float inputVal = fWeightMatrix0to1[o][i] * fWeights0[i];
           fWeights1[o] += inputVal;
        }
        fWeights1[o] = ActivationFnc(fWeights1[o]);
     }
     // layer 1 to 2
     for (int o=0; o<9; o++) {
        for (int i=0; i<11; i++) {
           float inputVal = fWeightMatrix1to2[o][i] * fWeights1[i];
           fWeights2[o] += inputVal;
        }
        fWeights2[o] = ActivationFnc(fWeights2[o]);
     }
     // layer 2 to 3
     for (int o=0; o<7; o++) {
        for (int i=0; i<10; i++) {
           float inputVal = fWeightMatrix2to3[o][i] * fWeights2[i];
           fWeights3[o] += inputVal;
        }
        fWeights3[o] = ActivationFnc(fWeights3[o]);
     }
     // layer 3 to 4
     for (int i=0; i<8; i++) {
       float inputVal = fWeightMatrix3to4[0][i] * fWeights3[i];
       fWeights4[0] += inputVal;
     }
     return OutputActivationFnc(fWeights4[0]);

  }

  // the classifier response
  // "inputValues" is an array of input values in the same order as the
  // variables given to the constructor
  // WARNING: inputVariables will be modified
  inline float GetMvaValue( float iV[7] ) const 
  {
    //Normalize input
    Transform_1( iV);
    
    return GetMvaValue__( iV );
  }
  
};
 
} // namespace SciFi
