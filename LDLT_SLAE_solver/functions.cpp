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

// Обработку возникающих нулевых блоков перенести в другое место - она должна произвестись ОДИН раз!!!
void calc_block(block* res, block* upper, block* lower, diagonal* diag) {
	if (upper == nullptr or lower == nullptr) return;
	if (res == nullptr) res = new block; 
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
					res->values[i * block_size + k] -= hsum256_pd_fast(_mm256_mul_pd(tmp, _mm256_load_pd(upper->values + k * block_size + j)));
				}
			}
		}
	}
	// Проверка на нулевой блок (если нулевой, то очищаем память) - важно, что память для блока выделялась именно в памяти экземпляра программы
	__m256d zero = _mm256_setzero_pd();
	for (int i = 0; i < block_size * block_size; i += 4) {
		__m256d vec = _mm256_load_pd(res->values + i);
		__m256d cmp = _mm256_cmp_pd(vec, zero, _CMP_NEQ_OQ);
		int mask = _mm256_movemask_pd(cmp);
		if (mask != 0) {
			return;
		}
	}
	delete res;
	res = nullptr;
}

//Есть ошибка при подсчете - не делили на диагональный прежде чем высчитывать следующие столбцы (или нет?) - Возможное решение - убрать домножение на диагональный в цикле по m
void calc_block_final(block* res, block* upper, diagonal* diag) {
	if (res == nullptr) return;
	#pragma omp parallel
	{
		int num_threads = omp_get_num_threads();
		int id = omp_get_thread_num();
		for (int i = id; i < block_size; i += num_threads) { // проход по ряду
			for (int j = 0; j < block_size; j += 4) {
				// Необходимо досчитать еще три столбца в самом блоке для того, чтобы далее эффективно использовать AVX, затем останется снова три столбца, которые придется считать без AVX
				for (int k = 1; k < 4; ++k) {
					for (int m = 0; m < k; ++m) {
						res->values[i * block_size + k + j] -= res->values[i * block_size + m + j] * upper->values[k * block_size + m + j];
					}
				}
				__m256d d = _mm256_load_pd(diag->values + j);
				_mm256_store_pd(res->values + i * block_size + j, _mm256_div_pd(_mm256_load_pd(res->values + i * block_size + j), d));
				__m256d low = _mm256_load_pd(res->values + i * block_size + j);
				__m256d tmp = _mm256_mul_pd(low, d);
				for (int k = j + 4; k < block_size; ++k) {
					res->values[i * block_size + k] -= hsum256_pd_fast(_mm256_mul_pd(tmp, _mm256_load_pd(upper->values + k * block_size + j)));
				}
			}
		}
	}
	// Проверка на нулевой блок (если нулевой, то очищаем память) - важно, что память для блока выделялась именно в памяти экземпляра программы
	__m256d zero = _mm256_setzero_pd();
	for (int i = 0; i < block_size * block_size; i += 4) {
		__m256d vec = _mm256_load_pd(res->values + i);
		__m256d cmp = _mm256_cmp_pd(vec, zero, _CMP_NEQ_OQ);
		int mask = _mm256_movemask_pd(cmp);
		if (mask != 0) {
			return;
		}
	}
	delete res;
	res = nullptr;
}


void calc_diag_block(block* res, block* other, diagonal* diag) {
	if (other == nullptr) return;
	#pragma omp parallel
	{
		int num_threads = omp_get_num_threads();
		int id = omp_get_thread_num();
		for (int i = id; i < block_size; i += num_threads) {
			for (int j = 0; j < block_size; j += 4) {
				__m256d d = _mm256_load_pd(diag->values + j);
				__m256d low = _mm256_load_pd(res->values + i * block_size + j);
				__m256d tmp = _mm256_mul_pd(low, d);
				for (int k = 0; k <= i; ++k) {
					res->values[i * block_size + k] -= hsum256_pd_fast(_mm256_mul_pd(tmp, _mm256_load_pd(res->values + k * block_size + j)));
				}
			}
		}
	}
}

void calc_diag_block_final(block* res, diagonal* diag) {
	for (int i = 0; i < block_size; ++i) {
		diag->values[i] = res->values[i * block_size + i];
	}
	for (int i = 0; i < block_size; i += 4) { // Сдвиг по субматрицам block_size x 4
		for (int j = 1; j < 4; ++j) { // Номера столбцов внутри субматрицы
			#pragma omp parallel
			{
				int num_threads = omp_get_num_threads();
				int id = omp_get_thread_num();
				for (int k = id + j + i; k < block_size; k += num_threads) { // Номера строк
					for (int m = 0; m < j; ++m) { // Проход по столбцам внутри субматрицы для дорасчета j-го стобца
						res->values[k * block_size + i + j] -= res->values[k * block_size + i + m] * res->values[(i + j) * block_size + i + m] / diag->values[i + m];
					}
				}
			}
		}
		for (int j = 0; j < 4; ++j) {
			diag->values[(i + j)] = res->values[(i + j) * block_size + i + j];
		}
		for (int j = 0; j < 3; ++j) {
			for (int k = j + 1; k < 4; ++k) {
				res->values[(i + k) * block_size + i + j] /= diag->values[i + j];
			}
		}
		#pragma omp parallel
		{
			int num_threads = omp_get_num_threads();
			int id = omp_get_thread_num();
			__m256d d = _mm256_load_pd(diag->values + i);
			for (int j = id + i + 4; j < block_size; j += num_threads) {
				_mm256_store_pd(res->values + j * block_size + i, _mm256_div_pd(_mm256_load_pd(res->values + j * block_size + i), d));
			}
			#pragma omp barrier
			for (int j = i + 4 + id; j < block_size; j += num_threads) {
				__m256d low = _mm256_load_pd(res->values + j * block_size + i);
				__m256d tmp = _mm256_div_pd(low, d);
				for (int k = j; k <= j; ++k) {
					res->values[j * block_size + k] -= hsum256_pd_fast(_mm256_mul_pd(tmp, _mm256_load_pd(res->values + k * block_size + i)));
				}
			}
		}
	}
}


void solve_L_SLAE(block* A, vect* b) {
	for (int i = 1; i < block_size; ++i) {
		for (int j = 0; j < i; ++j) {
			b->values[i] -= A->values[i * block_size + j] * b->values[j];
		}
	}
}

void solve_D_SLAE(diagonal* D, vect* b) {
	for (int i = 0; i < block_size; i += 4) {
		_mm256_store_pd(b->values + i, _mm256_div_pd(_mm256_load_pd(b->values + i), _mm256_load_pd(D->values + i)));
	}
}

void sub_block_mul_sol(block* A, vect* x, vect* b) {
	for (int i = 0; i < block_size; i += 4) {
		__m256d d = _mm256_load_pd(x->values + i);
		for (int j = 0; j < block_size; ++j) {
			b->values[j] -= hsum256_pd_fast(_mm256_mul_pd(d, _mm256_load_pd(A->values + block_size * j + i)));
		}
	}
}

void solve_LT_SLAE(block* A, vect* b) {
	for (int i = block_size - 2; i >= 0; --i) {
		for (int j = block_size - 1; j >= i + 1; --j) {
			b->values[i] -= A->values[j * block_size + i] * b->values[i];
		}
	}
}

void sub_block_T_mul_sol(block* A, vect* x, vect* b) {
	for (int n = 0; n < block_size; ++n) {
		for (int i = 0; i < block_size; ++i) {
			b->values[i] -= A->values[i * block_size + n] * x->values[i];
		}
	}
}