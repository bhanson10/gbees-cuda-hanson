// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#ifndef MATHS_H
#define MATHS_H

#include "config.h"

/** 
 * @brief Invert a matrix
 * 
 * @param matrix the matrix to invert
 * @param inverse [output] the inverse of matrix to fill in
 * @param size of the matrices
 */
__host__  void invertMatrix(float* matrix, float* inverse, int size);

/**
 * @brief Multiply a square matrix by a vector
 * 
 * @param matrix the matrix
 * @param vector the vector
 * @param result [output] the output vector
 * @param size the size of the matrix
 */
__device__ void multiplyMatrixVector(float* matrix, float* vector, float* result, int size);

/**
 * @brief Compute dot product
 * @param vec1 first vector
 * @param vec2 second vector
 * @param size vector size
 * @return the dot product
 */
__device__ float computeDotProduct(float* vec1, float* vec2, int size);

#endif