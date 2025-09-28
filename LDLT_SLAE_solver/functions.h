#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <immintrin.h>
#include <omp.h>
#include <vector>

#define block_size 32

struct alignas(32) block {
	double values[block_size * block_size] = {};
};

struct alignas(32) diagonal {
	double values[block_size] = {};
};

struct alignas(32) vect {
	double values[block_size] = {};
};

struct matrix {
	block** blocks;
	diagonal** diagonals;
	int size;
};

matrix* read_matrix(std::string);

void calc_block(block*, block*, block*, diagonal*);

int calc_block_final(block*, block*, diagonal*);

void calc_diag_block(block*, block*, diagonal*);

void calc_diag_block_final(block*, diagonal*);

void solve_L_SLAE(block*, vect*);

void solve_D_SLAE(diagonal*, vect*);

void sub_block_mul_sol(block*, vect*, vect*);

void solve_LT_SLAE(block*, vect*);

void sub_block_T_mul_sol(block*, vect*, vect*);

double check_solution(std::string, std::string, std::string);

static inline double hsum256_pd_fast(__m256d v) {
	__m256d t1 = _mm256_permute2f128_pd(v, v, 0x1); // [x2,x3,x0,x1]
	__m256d t2 = _mm256_add_pd(v, t1);              // [x0+x2, x1+x3, ...]
	__m256d t3 = _mm256_permute_pd(t2, 0b0101);     // поменять местами внутри 128-бит
	__m256d t4 = _mm256_add_pd(t2, t3);             // [sum,sum,sum,sum]
	return _mm_cvtsd_f64(_mm256_castpd256_pd128(t4));
}
