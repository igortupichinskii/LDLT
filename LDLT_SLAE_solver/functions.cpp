#include "functions.h"


matrix* read_matrix(std::string filename) {
	std::ifstream file(filename);
	if (!file.is_open()){
		return nullptr;
	}
	else {
		matrix* m = new matrix;
		std::string buf;
		int SIZE, NONZEROS, size_block_matrix;
		while (!file.eof()) {
			if (!std::getline(file, buf)) return nullptr;
			if (buf[0] != '%') {
				std::istringstream iss(buf);
				iss >> SIZE >> NONZEROS;
				size_block_matrix = SIZE / block_size + (int)(!!(SIZE / block_size));
				(*m).blocks = new block* [size_block_matrix * size_block_matrix];
				(*m).diagonals = new diagonal * [size_block_matrix];
				for (int i = 0; i < size_block_matrix * size_block_matrix; i++) {
					m->blocks[i] = nullptr;
				}
				for (int i = 0; i < size_block_matrix; ++i) {
					m->diagonals[i] = new diagonal;
				}
				(*m).size = size_block_matrix;
				std::cout << "Matrix memory initialized" << std::endl;
				break;
			}
		}
		int col, row, b_col, b_row, row_in_block, col_in_block, block_array_ind, index_in_block;
		double val;
		while (!file.eof()) {
			file >> row >> col >> val;
			--row;
			--col;
			b_row = row / block_size;
			b_col = col / block_size;
			row_in_block = row % block_size;
			col_in_block = col % block_size;
			block_array_ind = b_col * size_block_matrix + b_row;
			index_in_block = row_in_block * block_size + col_in_block;
			if (m->blocks[block_array_ind]) {
				m->blocks[block_array_ind]->values[index_in_block] = val;
			}
			else {
				m->blocks[block_array_ind] = new block;
				m->blocks[block_array_ind]->values[index_in_block] = val;
			}
			if (row == col) {
				m->diagonals[b_row]->values[row_in_block] = val;
			}
		}
		return m;
	}
}


void calc_block(block* res, block* upper, block* lower, diagonal* diag) {
	#pragma omp parallel
	{
		int num_threads = omp_get_num_threads();
		int id = omp_get_thread_num();
		for (int i = id; i < block_size; i += num_threads) { // проход по ряду
			for (int j = 0; j < block_size; j += 4) { // Проход по кускам ряда длины 4
				__m256d low = _mm256_load_pd(lower->values + block_size * i + j); // Нижняя матрица, общая для ряда
				__m256d d = _mm256_load_pd(diag->values + j); // Диагональная
				__m256d tmp = _mm256_mul_pd(low, d);
				for (int k = 0; k < block_size; ++k) {
					res->values[i * block_size + k] += hsum256_pd_fast(_mm256_mul_pd(tmp, _mm256_load_pd(upper->values + )));
				}
				/*__m256d up = _mm256_load_pd(upper->values + block_size * i + j);
				__m256d low = _mm256_load_pd(lower->values + block_size * i + j);
				__m256d r = _mm256_load_pd(res->values + block_size * i + j);
				__m256d d = _mm256_load_pd(diag->values + j);
				__m256d tmp = _mm256_mul_pd(up, low);
				__m256d f = _mm256_fmadd_pd(tmp, d, r);
				_mm256_store_pd(res->values + block_size * i + j, f);*/
			}
		}
	}
}

void calc_block_final(block* res, block* upper, diagonal* diag) {
	#pragma omp parallel
	{
		int num_threads = omp_get_num_threads();
		int id = omp_get_thread_num();

	}
}