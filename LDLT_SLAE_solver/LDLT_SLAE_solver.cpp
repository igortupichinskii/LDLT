// LDLT_SLAE_solver.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//
#include "functions.h"
#include <iostream>
#include "mpi.h"
#include <omp.h>
#include <fstream>
#define TAG_READY 100
#define TAG_TASK 101
#define TAG_TASK_LAST 102
#define TAG_RESULT 103
#define TAG_WAIT 104
#define TAG_GET_DATA 105
#define TAG_DATA 106
#define TAG_GET_DIAG 107
#define TAG_DIAG 108
#define TAG_NEW_COLUMN 109

struct TaskHdr {
    int col_j;
    int row_i;
    int is_null;
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

    MPI_Datatype MPI_BLOCK, MPI_DIAG;
    MPI_Type_contiguous(block_size * block_size, MPI_DOUBLE, &MPI_BLOCK);
    MPI_Type_commit(&MPI_BLOCK);
    MPI_Type_contiguous(block_size, MPI_DOUBLE, &MPI_DIAG);
    MPI_Type_commit(&MPI_DIAG);


    matrix* m = nullptr;
    int nb = 0;
    if (rank == 0) { // Главный процесс - должен раздавать задачи другим процессам, также занимается вычислением диагональных блоков матрицы
        m = read_matrix("C:/Users/itupi/OneDrive/Документы/OS/igortupichinskii-project/LDLT/494_bus.mtx");
        nb = m->size;
        MPI_Barrier(MPI_COMM_WORLD);
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
                        int hdr[3];
                        hdr[1] = column_n;
                        hdr[0] = ++last_index;
                        int block_index = hdr[0] * nb + hdr[1];
                        hdr[2] = (m->blocks[block_index] == nullptr) ? 0 : 1;
                        MPI_Send(hdr, 3, MPI_INT, st.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD); //Отправка индексов и пустоты блока
                        if (hdr[2]) {
                            send_block(m->blocks[block_index], st.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD);
                        }
                        tasks_in_proceed++;
                    }
                    else { //Пока воркер должен стоять в простое (вызвать барьер) - это должны получить все воркеры
                        MPI_Send(NULL, 1, MPI_BYTE, st.MPI_SOURCE, TAG_WAIT, MPI_COMM_WORLD);
                    }
                }
                else {
                    if (st.MPI_TAG == TAG_GET_DATA) { // Это значит, что он хочет получить новые блоки
                        int hdr[3]; // Столбец, две строки в порядке возрастания
                        MPI_Recv(hdr, 3, MPI_INT, st.MPI_SOURCE, TAG_GET_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        if (m->blocks[hdr[1] * nb + hdr[0]] == nullptr || m->blocks[hdr[2] * nb + hdr[0]] == nullptr) { //Какой-то из блоков пустой - значит, надо пройти дальше
                            int new_col = hdr[0];
                            while (new_col < hdr[1]) {
                                if (m->blocks[hdr[1] * nb + new_col] && m->blocks[hdr[2] * nb + new_col]) {
                                    break;
                                }
                                ++new_col;
                            }
                            if (new_col == hdr[1]) {
                                send_block(m->blocks[new_col * nb + new_col], st.MPI_SOURCE, TAG_DIAG, MPI_COMM_WORLD);
                                send_diag(m->diagonals[new_col], st.MPI_SOURCE, TAG_DIAG, MPI_COMM_WORLD);
                            }
                            else {
                                MPI_Send(&new_col, 1, MPI_INT, st.MPI_SOURCE, TAG_NEW_COLUMN, MPI_COMM_WORLD);
                                send_block(m->blocks[hdr[1] * nb + new_col], st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                                send_block(m->blocks[hdr[2] * nb + new_col], st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                                send_diag(m->diagonals[new_col], st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                            }
                        }
                        else {
                            send_block(m->blocks[hdr[1] * nb + hdr[0]], st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                            send_block(m->blocks[hdr[2] * nb + hdr[0]], st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                            send_diag(m->diagonals[hdr[0]], st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                        }
                    }
                    else {
                        if (st.MPI_TAG == TAG_RESULT) { // Это значит, что он досчитал блок
                            int hdr[3]; //индексы блока и заполненность
                            MPI_Recv(hdr, 3, MPI_INT, st.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            if (hdr[2]) {
                                if (!(m->blocks[hdr[0] * nb + hdr[1]])) m->blocks[hdr[0] * nb + hdr[1]] = new block;
                                recv_block(m->blocks[hdr[0] * nb + hdr[1]], st.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD);
                            }
                            tasks_in_proceed--;
                        }
                        else {
                            if (st.MPI_TAG == TAG_GET_DIAG) {
                                int col;
                                MPI_Recv(&col, 1, MPI_INT, st.MPI_SOURCE, TAG_GET_DIAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                send_block(m->blocks[col * nb + col], st.MPI_SOURCE, TAG_DIAG, MPI_COMM_WORLD);
                                send_diag(m->diagonals[col], st.MPI_SOURCE, TAG_DIAG, MPI_COMM_WORLD);
                            }
                        }
                    }
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

    }
    else { //Workers
        MPI_Barrier(MPI_COMM_WORLD);
        int dummy = 1;
        block* upper= new block;
        block* diag_b = new block;
        block* lower = new block;
        block* work = new block;
        diagonal* diag = new diagonal;
        while (true) {
            MPI_Send(&dummy, 1, MPI_INT, 0, TAG_READY, MPI_COMM_WORLD);
            MPI_Status st;
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
            if (st.MPI_TAG == TAG_WAIT) {
                MPI_Recv(NULL, 0, MPI_BYTE, 0, TAG_WAIT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Barrier(MPI_COMM_WORLD);
            }
            else {
                if (st.MPI_TAG == TAG_TASK) {
                    int hdr[3];
                    MPI_Recv(hdr, 3, MPI_INT, 0, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (hdr[2]) {
                        recv_block(work, 0, TAG_TASK, MPI_COMM_WORLD);
                    }
                    int b_col = hdr[1];
                    int b_row = hdr[0];
                    int col = 0;
                    while (col < b_col) {
                        int hdr[3];
                        hdr[0] = col;
                        hdr[1] = b_col;
                        hdr[2] = b_row;
                        MPI_Send(hdr, 3, MPI_INT, 0, TAG_GET_DATA, MPI_COMM_WORLD);
                        MPI_Status st_data;
                        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &st_data);
                        switch (st_data.MPI_TAG) {
                        case TAG_DIAG:
                            recv_block(diag_b, 0, TAG_DIAG, MPI_COMM_WORLD);
                            recv_diag(diag, 0, TAG_DIAG, MPI_COMM_WORLD);
                            col = b_col;
                            calc_block_final(work, diag_b, diag);
                            break;
                        case TAG_NEW_COLUMN:
                            MPI_Recv(&col, 1, MPI_INT, 0, TAG_NEW_COLUMN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            recv_block(upper, 0, TAG_DATA, MPI_COMM_WORLD);
                            recv_block(lower, 0, TAG_DATA, MPI_COMM_WORLD);
                            recv_diag(diag, 0, TAG_DATA, MPI_COMM_WORLD);
                            calc_block(work, upper, lower, diag);
                            break;
                        case TAG_DATA:
                            recv_block(upper, 0, TAG_DATA, MPI_COMM_WORLD);
                            recv_block(lower, 0, TAG_DATA, MPI_COMM_WORLD);
                            recv_diag(diag, 0, TAG_DATA, MPI_COMM_WORLD);
                            calc_block(work, upper, lower, diag);
                            break;
                        default:
                            break;
                        }
                        col++;
                    }
                    hdr[0] = b_row;
                    hdr[1] = b_col;
                    if (col == b_col) {
                        MPI_Send(&col, 1, MPI_INT, 0, TAG_GET_DIAG, MPI_COMM_WORLD);
                        recv_block(diag_b, 0, TAG_DIAG, MPI_COMM_WORLD);
                        recv_diag(diag, 0, TAG_DIAG, MPI_COMM_WORLD);
                        hdr[2] = calc_block_final(work, diag_b, diag);
                    }
                    MPI_Send(&hdr, 3, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
                    if (hdr[2]) {
                        send_block(work, 0, TAG_RESULT, MPI_COMM_WORLD);
                    }
                }
            }
        }
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
