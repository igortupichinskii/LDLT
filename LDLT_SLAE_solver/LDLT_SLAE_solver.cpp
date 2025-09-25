// LDLT_SLAE_solver.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//
#include "functions.h"
#include <iostream>
#include "mpi.h"
#include <omp.h>
#define TAG_READY 100
#define TAG_TASK 101
#define TAG_TASK_LAST 102
#define TAG_RESULT 103
#define TAG_STOP 104
#define TAG_GET_DATA 105

struct TaskHdr {
    int col_j;
    int row_i;
};

static inline void send_block(const block* b, int dst, int tag, MPI_Comm comm) {
    MPI_Send(const_cast<double*>(b->values), block_size * block_size, MPI_DOUBLE, dst, tag, comm);
}
static inline void recv_block(block* b, int src, int tag, MPI_Comm comm) {
    MPI_Recv(b->values, block_size * block_size, MPI_DOUBLE, src, tag, comm, MPI_STATUS_IGNORE);
}
static inline void send_diag(const diagonal* d, int dst, int tag, MPI_Comm comm) {
    MPI_Send(const_cast<double*>(d->values), block_size, MPI_DOUBLE, dst, tag, comm);
}
static inline void recv_diag(diagonal* d, int src, int tag, MPI_Comm comm) {
    MPI_Recv(d->values, block_size, MPI_DOUBLE, src, tag, comm, MPI_STATUS_IGNORE);
}

int main(int argc, char *argv[])
{
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    matrix* m = nullptr;
    int nb = 0;
    if (rank == 0) { // Главный процесс - должен раздавать задачи другим процессам, также занимается вычислением диагональных блоков матрицы
        m = read_matrix("C:/Users/itupi/OneDrive/Документы/OS/igortupichinskii-project/LDLT/494_bus.mtx");
        nb = m->size;
    }
    MPI_Bcast(&nb, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (int column_n = 0; column_n < nb; ++column_n) {
            // Раскладываем диагональный
            for (int i = 0; i < column_n; ++i) {
                calc_diag_block(m->blocks[column_n * nb + column_n], m->blocks[column_n * nb + i], m->diagonals[i]);
            }
            calc_diag_block_final(m->blocks[column_n * nb + column_n], m->diagonals[column_n]);
            //После этого диагональный элемент столбца готов - можно раздавать блоки
            int last_index = column_n;
            int tasks_in_proceed=0;
            while (last_index < nb || tasks_in_proceed) {
                MPI_Status st;
                MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
                if (st.MPI_TAG == TAG_READY) { // Это значит, что отправивший сообщение процесс сейчас готов получить задание
                    int dummy;
                    MPI_Recv(&dummy, 1, MPI_INT, st.MPI_SOURCE, TAG_READY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (last_index < nb) {
                        TaskHdr hd{};
                        hd.col_j = column_n;
                        hd.row_i = ++last_index;
                        
                    }
                }
                else {
                    if (st.MPI_TAG == TAG_GET_DATA) { // Это значит, что он хочет получить новые блоки

                    }
                    else {
                        if (st.MPI_TAG == TAG_RESULT) { // Это значит, что он досчитал блок

                        }
                        else {

                        }
                    }
                }
            }
        }

    }
    else {

    }
    MPI_Finalize();
}

// Запуск программы: CTRL+F5 или меню "Отладка" > "Запуск без отладки"
// Отладка программы: F5 или меню "Отладка" > "Запустить отладку"

// Советы по началу работы 
//   1. В окне обозревателя решений можно добавлять файлы и управлять ими.
//   2. В окне Team Explorer можно подключиться к системе управления версиями.
//   3. В окне "Выходные данные" можно просматривать выходные данные сборки и другие сообщения.
//   4. В окне "Список ошибок" можно просматривать ошибки.
//   5. Последовательно выберите пункты меню "Проект" > "Добавить новый элемент", чтобы создать файлы кода, или "Проект" > "Добавить существующий элемент", чтобы добавить в проект существующие файлы кода.
//   6. Чтобы снова открыть этот проект позже, выберите пункты меню "Файл" > "Открыть" > "Проект" и выберите SLN-файл.
