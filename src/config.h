// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#ifndef CONFIG_H
#define CONFIG_H

/** Grid dimension */
// #define DIM 3 // Lorenz3D
// #define DIM 4 // PCR3BP
#define DIM 6 // CR3BP
// #define DIM 6 // Lorenz6D

/** Number of blocks */
// #define BLOCKS 96 // Lorenz3D
// #define BLOCKS 320 // PCR3BP
#define BLOCKS 320 // CR3BP
// #define BLOCKS 320 // Lorenz6D

/** Number of threads per block */
// #define THREADS_PER_BLOCK 256 // Lorenz3D
// #define THREADS_PER_BLOCK 512 // PCR3BP
#define THREADS_PER_BLOCK 512 // CR3BP
// #define THREADS_PER_BLOCK 512 // Lorenz6D

/** Number of cells that process one thread */
// #define CELLS_PER_THREAD 1 // Lorenz3D
// #define CELLS_PER_THREAD 9 // PCR3BP
#define CELLS_PER_THREAD 2000 // CR3BP
// #define CELLS_PER_THREAD 350 // Lorenz6D

/** Enable logs (comment out to disable logs) */
#define ENABLE_LOG

/** Size of the hashtable with respect the maximum number of cells*/
#define HASH_TABLE_RATIO 2

/**  Left uncommented for single precission in the snapshots, comment it for double precision */
#define SINGLE_PRECISION_SNAPSHOTS

#endif