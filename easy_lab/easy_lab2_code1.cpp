/**
 * @file easy_lab2_code1.cpp
 * @author Du Jiajun (Dujiajun@bupt.edu.cn)
 * @date 2023-12-13
 * 
 * @copyright BUPT-OS easy_lab2
 * 
 */
#include "multiply.h"
#include <thread>
#include <vector>

// 加速方法: 使用了多线程、cache两种加速方法
// 服务器测试结果:
//        matrix size                    average time           GFlops
// (512 * 512)   x (512 * 512)          18953.697800 us         14.162696
// (1024 * 1024) x (1024 * 1024)        76267.036800 us         28.157429
// (2048 * 2048) x (2048 * 2048)        433899.779000 us        39.594095
// (2560 * 2560) x (2560 * 2560)        754310.254800 us        44.483595
// (3072 * 3072) x (3072 * 3072)        1186699.031400 us       48.859953

using std::vector, std::thread, std::min;

/*BLOCK_SIZE*/
#define N_BLOCK 12
#define M_BLOCK 12
#define P_BLOCK P
#define L_BLOCK 4

// buffer2 need to be used between different threads
double buffer2[M][P];

void compute4x4(double *b1, double *b2, double *r0, int col1_count)
{
    double
        r_00_reg = 0.0, r_01_reg = 0.0, r_02_reg = 0.0, r_03_reg = 0.0,
        r_10_reg = 0.0, r_11_reg = 0.0, r_12_reg = 0.0, r_13_reg = 0.0,
        r_20_reg = 0.0, r_21_reg = 0.0, r_22_reg = 0.0, r_23_reg = 0.0,
        r_30_reg = 0.0, r_31_reg = 0.0, r_32_reg = 0.0, r_33_reg = 0.0,
        b1_0_reg, b1_1_reg, b1_2_reg, b1_3_reg,
        b2_0_reg, b2_1_reg, b2_2_reg, b2_3_reg;

    int p = 0;
    for(; p < col1_count; ++p)
    {
        b1_0_reg = *(b1++);
        b1_1_reg = *(b1++);
        b1_2_reg = *(b1++);
        b1_3_reg = *(b1++);

        b2_0_reg = *(b2++);
        b2_1_reg = *(b2++);
        b2_2_reg = *(b2++);
        b2_3_reg = *(b2++);

        r_00_reg += b1_0_reg * b2_0_reg;
        r_01_reg += b1_0_reg * b2_1_reg;
        r_02_reg += b1_0_reg * b2_2_reg;
        r_03_reg += b1_0_reg * b2_3_reg;

        r_10_reg += b1_1_reg * b2_0_reg;
        r_11_reg += b1_1_reg * b2_1_reg;
        r_12_reg += b1_1_reg * b2_2_reg;
        r_13_reg += b1_1_reg * b2_3_reg;

        r_20_reg += b1_2_reg * b2_0_reg;
        r_21_reg += b1_2_reg * b2_1_reg;
        r_22_reg += b1_2_reg * b2_2_reg;
        r_23_reg += b1_2_reg * b2_3_reg;

        r_30_reg += b1_3_reg * b2_0_reg;
        r_31_reg += b1_3_reg * b2_1_reg;
        r_32_reg += b1_3_reg * b2_2_reg;
        r_33_reg += b1_3_reg * b2_3_reg;
    }

    double *r1 = r0 + P;
    double *r2 = r1 + P;
    double *r3 = r2 + P;

    *(r0) += r_00_reg; *(r0 + 1) += r_01_reg; *(r0 + 2) += r_02_reg; *(r0 +3) += r_03_reg;
    *(r1) += r_10_reg; *(r1 + 1) += r_11_reg; *(r1 + 2) += r_12_reg; *(r1 +3) += r_13_reg;
    *(r2) += r_20_reg; *(r2 + 1) += r_21_reg; *(r2 + 2) += r_22_reg; *(r2 +3) += r_23_reg;
    *(r3) += r_30_reg; *(r3 + 1) += r_31_reg; *(r3 + 2) += r_32_reg; *(r3 +3) += r_33_reg;
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