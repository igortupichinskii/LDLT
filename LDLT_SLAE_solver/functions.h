#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <immintrin.h>
#include <omp.h>
#include <vector>
#include <random>
#include <iomanip>

#define block_size 32

struct alignas(32) block {
	double values[block_size * block_size] = {};
	void* operator new(size_t size) {
		return _aligned_malloc(size, 32);
	}

	void operator delete(void* ptr) {
		_aligned_free(ptr);
	}
};

struct alignas(32) diagonal {
	double values[block_size] = {};

	void* operator new(size_t size) {
		return _aligned_malloc(size, 32);
	}

	void operator delete(void* ptr) {
		_aligned_free(ptr);
	}
};

struct alignas(32) vect {
	double values[block_size] = {};

	void* operator new(size_t size) {
		return _aligned_malloc(size, 32);
	}

	void operator delete(void* ptr) {
		_aligned_free(ptr);
	}
};

struct matrix {
	block** blocks;
	diagonal** diagonals;
	int size;
};

matrix* read_matrix(std::string);

void calc_block(block*, block*, block*, diagonal*);

void calc_block_non_parallel(block*, block*, block*, diagonal*);

int calc_block_final(block*, block*, diagonal*);

int calc_block_final_non_parallel(block*, block*, diagonal*);

void calc_diag_block(block*, block*, diagonal*);

void calc_diag_block_non_parallel(block*, block*, diagonal*);

void calc_diag_block_final(block*, diagonal*);

int calc_diag_block_final_non_parallel(block*, diagonal*);

void solve_L_SLAE(block*, vect*);

void solve_D_SLAE(diagonal*, vect*);

void solve_D_SLAE_non_parallel(diagonal*, vect*);

void sub_block_mul_sol(block*, vect*, vect*);

void sub_block_mul_sol_non_parallel(block*, vect*, vect*);

void solve_LT_SLAE(block*, vect*);

void sub_block_T_mul_sol(block*, vect*, vect*);

double check_solution(std::string, std::string, std::string);

double random_double(double, double);

vect** random_vector_generation(int);

void write_decomp_to_file(matrix*, std::string);

static inline double hsum256_pd_fast(__m256d v) {
	__m256d t1 = _mm256_permute2f128_pd(v, v, 0x1); // [x2,x3,x0,x1]
	__m256d t2 = _mm256_add_pd(v, t1);              // [x0+x2, x1+x3, ...]
	__m256d t3 = _mm256_permute_pd(t2, 0b0101);     // поменять местами внутри 128-бит
	__m256d t4 = _mm256_add_pd(t2, t3);             // [sum,sum,sum,sum]
	return _mm_cvtsd_f64(_mm256_castpd256_pd128(t4));
}
