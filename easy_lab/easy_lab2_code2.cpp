/**
 * @file easy_lab2_code2.cpp
 * @author Du Jiajun (Dujiajun@bupt.edu.cn)
 * @date 2023-12-13
 * 
 * @copyright BUPT-OS easy_lab2
 * 
 */
#include "multiply.h"
#include <thread>
#include <mmintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <emmintrin.h>
#include <vector>

// Ways to speed up matrix multiplication:
//      Higher Cache Hit Rate
//      Parallel Programming
//      SSE3 Instruction Set
//
// Test results on the server:
//        matrix size                    average time           GFlops
// (512 * 512)   x (512 * 512)          17766.459400 us         15.109
// (1024 * 1024) x (1024 * 1024)        62070.710800 us         34.597375
// (2048 * 2048) x (2048 * 2048)        342196.942600 us        49.152825
// (2560 * 2560) x (2560 * 2560)        680797.943200 us        49.286917
// (3072 * 3072) x (3072 * 3072)        1071400.119000 us       54.118025

using std::vector, std::thread, std::min;

/*BLOCK_SIZE*/
#define N_BLOCK 12
#define M_BLOCK 12
#define P_BLOCK P
#define L_BLOCK 4

// buffer2 need to be used between different threads
double buffer2[M][P];

typedef union {
    __m128d vector;
    double data[2];
} vector_register;

void compute4x4(double *b1, double *b2, double *r0, int col1_count)
{
    vector_register
        r_00_10, r_01_11, r_02_12, r_03_13,
        r_20_30, r_21_31, r_22_32, r_23_33,
        b1_0_1, b1_2_3,
        b2_0, b2_1, b2_2, b2_3;

    r_00_10.vector = _mm_setzero_pd();
    r_01_11.vector = _mm_setzero_pd();
    r_02_12.vector = _mm_setzero_pd();
    r_03_13.vector = _mm_setzero_pd();
    r_20_30.vector = _mm_setzero_pd();
    r_21_31.vector = _mm_setzero_pd();
    r_22_32.vector = _mm_setzero_pd();
    r_23_33.vector = _mm_setzero_pd();

    int p = 0;
    for(; p < col1_count; ++p)
    {
        b1_0_1.vector = _mm_load_pd(b1);
        b1_2_3.vector = _mm_load_pd((b1 + 2));
        b1 += 4;

        b2_0.vector = _mm_loaddup_pd((b2));
        b2_1.vector = _mm_loaddup_pd((b2 + 1));
        b2_2.vector = _mm_loaddup_pd((b2 + 2));
        b2_3.vector = _mm_loaddup_pd((b2 + 3));
        b2 += 4;

        /* First row and second rows */
        r_00_10.vector += b1_0_1.vector * b2_0.vector;
        r_01_11.vector += b1_0_1.vector * b2_1.vector;
        r_02_12.vector += b1_0_1.vector * b2_2.vector;
        r_03_13.vector += b1_0_1.vector * b2_3.vector;

        /* Third and fourth rows */
        r_20_30.vector += b1_2_3.vector * b2_0.vector;
        r_21_31.vector += b1_2_3.vector * b2_1.vector;
        r_22_32.vector += b1_2_3.vector * b2_2.vector;
        r_23_33.vector += b1_2_3.vector * b2_3.vector;
    }

    // store to RAM
    double *r1 = r0 + P;
    double *r2 = r1 + P;
    double *r3 = r2 + P;
    // first line
    *r0 += r_00_10.data[0];         *(r0 + 1) += r_01_11.data[0];
    *(r0 + 2) += r_02_12.data[0];   *(r0 + 3) += r_03_13.data[0];
    // second line
    *r1 += r_00_10.data[1];         *(r1 + 1) += r_01_11.data[1];
    *(r1 + 2) += r_02_12.data[1];   *(r1 + 3) += r_03_13.data[1];
    // third line
    *r2 += r_20_30.data[0];         *(r2 + 1) += r_21_31.data[0];
    *(r2 + 2) += r_22_32.data[0];   *(r2 + 3) += r_23_33.data[0];
    // fourth line
    *r3 += r_20_30.data[1];         *(r3 + 1) += r_21_31.data[1];
    *(r3 + 2) += r_22_32.data[1];   *(r3 + 3) += r_23_33.data[1];
}

void buffer_matrix1(double *src, double *dst, int row1_count, int col1_count)
{
    // N % 4 == 0
    int row1_block_count = row1_count / L_BLOCK;
    double *src0, *src1, *src2, *src3;
    while(row1_block_count--) {
        src0 = src;
        src1 = src + M;
        src2 = src1 + M;
        src3 = src2 + M;
        src = src3 + M;
        for(int col1 = 0; col1 < col1_count; ++col1) {
            *(dst++) = *(src0++);
            *(dst++) = *(src1++);
            *(dst++) = *(src2++);
            *(dst++) = *(src3++);
        }
    }
}

void block_multiplication(double *matrix1, double *result_matrix,
                          int row1, int row1_count, int col1, int col1_count)
{
    // First, buffer matrix1 to continuous memory
    double buffer1[row1_count * col1_count];
    double *dst = (double *) buffer1;
    double *src = (double *) matrix1 + row1 * M + col1;
    buffer_matrix1(src, dst, row1_count, col1_count);

    const int row1_max = row1 + row1_count, const_gap1 = col1_count * L_BLOCK;
    int col2, gapb1 = 0, gapb2 = col1 * P;
    for(; row1 < row1_max; row1 += L_BLOCK, gapb1 += const_gap1) {
        for(col2 = 0; col2 < P; col2 += L_BLOCK) {
            compute4x4((double *) buffer1 + gapb1,
                       (double *) buffer2 + gapb2 + col2 * col1_count,
                       result_matrix + row1 * P + col2, col1_count);
        }
    }
}

void thread_task(double *matrix1, double *result_matrix, int row1, int row1_count)
{
    // col1 == row2
    int col1, col1_count;
    for(col1 = 0; col1 < M; col1 += M_BLOCK) {
        col1_count = min(M_BLOCK, M - col1);
        block_multiplication(matrix1, result_matrix, row1, row1_count, col1, col1_count);
    }
}

void buffer_matrix2(double *src, double *dst, int row2_count)
{
    double *row2_pointers[row2_count];
    int index, col2;
    for(index = 0; index < row2_count; ++index) {
        row2_pointers[index] = src + index * P;
    }
    // P % 4 == 0
    for(col2 = 0; col2 < P; col2 += L_BLOCK) {
        for(index = 0; index < row2_count; ++index) {
            *(dst++) = *(row2_pointers[index]++);
            *(dst++) = *(row2_pointers[index]++);
            *(dst++) = *(row2_pointers[index]++);
            *(dst++) = *(row2_pointers[index]++);
        }
    }
}

void matrix_multiplication(double matrix1[N][M], double matrix2[M][P],
                           double result_matrix[N][P])
{
    // First, buffer matrix2 in continuous memory
    int row2, row2_count;
    double *dst = (double *) buffer2;
    double *src = (double *) matrix2;
    int length;
    for(row2 = 0; row2 < M; row2 += M_BLOCK) {
        row2_count = min(M_BLOCK, M - row2);
        buffer_matrix2(src, dst, row2_count);
        length = row2_count * P;
        src += length;
        dst += length;
    }

    vector<thread>pool;
    // out blocks
    int row1, row1_count;
    for(row1 = 0; row1 < N; row1 += N_BLOCK) {
        row1_count = min(N_BLOCK, N - row1);
        // create thread_task
        pool.push_back(thread(&thread_task, (double *) matrix1,
                             (double *) result_matrix, row1, row1_count));
    }
    for(auto &t: pool) {
        t.join();
    }
}