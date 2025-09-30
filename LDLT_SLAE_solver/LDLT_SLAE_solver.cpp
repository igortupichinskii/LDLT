// LDLT_SLAE_solver.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//
#include <chrono>
#include <iomanip>
#include <windows.h>
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
#define TAG_STOP 200
#define TAG_START 500

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
static inline void send_vect(const vect* d, int dst, int tag, MPI_Comm comm) {
    MPI_Send(const_cast<double*>(d->values), block_size, MPI_DOUBLE, dst, tag, comm);
}
static inline void recv_vect(vect* d, int src, int tag, MPI_Comm comm) {
    MPI_Recv(d->values, block_size, MPI_DOUBLE, src, tag, comm, MPI_STATUS_IGNORE);
}

void set_processor_affinity(int rank) {
    DWORD_PTR processAffinityMask = 0;
    DWORD_PTR systemAffinityMask = 0;

    HANDLE hProcess = GetCurrentProcess();
    GetProcessAffinityMask(hProcess, &processAffinityMask, &systemAffinityMask);

    // Распределяем ядра по процессам: 0,2,4,6,8,10
    int core = rank * 2;
    DWORD_PTR newMask = 1ULL << core;

    SetProcessAffinityMask(hProcess, newMask);
    std::cout << "Process " << rank << " bound to core " << core << std::endl;
}

std::string convert_to_forward_slashes(const std::string& path) {
    std::string result = path;
    for (auto& c : result) {
        if (c == '\\') c = '/';
    }
    return result;
}

int main(int argc, char *argv[]) //Только путь до файла матрицы
{
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    set_processor_affinity(rank);

    MPI_Datatype MPI_BLOCK, MPI_DIAG;
    MPI_Type_contiguous(block_size * block_size, MPI_DOUBLE, &MPI_BLOCK);
    MPI_Type_commit(&MPI_BLOCK);
    MPI_Type_contiguous(block_size, MPI_DOUBLE, &MPI_DIAG);
    MPI_Type_commit(&MPI_DIAG);


    matrix* m = nullptr;
    int nb = 0;

    std::chrono::high_resolution_clock::time_point start_time, end_time;

    if (rank == 0) { // Главный процесс - должен раздавать задачи другим процессам, также занимается вычислением диагональных блоков матрицы
        m = read_matrix(convert_to_forward_slashes(argv[1]));
        if (m == nullptr) {
            std::cerr << "Failed to read matrix" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        nb = m->size;
        write_decomp_to_file(m, "M_before_LDLT.txt");
        start_time = std::chrono::high_resolution_clock::now();
        std::cout << "Matrix read\n";
    }
    MPI_Bcast(&nb, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (int column_n = 0; column_n < nb; ++column_n) { //column_n - индекс столбца блочной матрицы, расчет разложения которого сейчас ведется
            // Раскладываем диагональный
            for (int i = 0; i < column_n; ++i) {
                calc_diag_block_non_parallel(m->blocks[column_n * nb + column_n], m->blocks[column_n + i * nb], m->diagonals[i]);
            }
            auto is_there_nan = calc_diag_block_final_non_parallel(m->blocks[column_n * nb + column_n], m->diagonals[column_n]);
            if (is_there_nan == -1) {
                std::cerr << "Nan occured in diag block " << column_n  << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            //После этого диагональный элемент столбца готов - можно раздавать блоки
            int last_index = column_n;
            int tasks_in_proceed=0;
            std::vector<int> waiting_workers;
            while (last_index < nb - 1 || tasks_in_proceed) {
                MPI_Status st;
                MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
                if (st.MPI_TAG == TAG_READY) { // Это значит, что отправивший сообщение процесс сейчас готов получить задание
                    int dummy;
                    MPI_Recv(&dummy, 1, MPI_INT, st.MPI_SOURCE, TAG_READY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (last_index < nb - 1) {
                        int hdr[3];
                        hdr[1] = column_n;
                        hdr[0] = ++last_index;
                        int block_index = hdr[0] + hdr[1] * nb;
                        hdr[2] = (m->blocks[block_index] == nullptr) ? 0 : 1;
                        MPI_Send(hdr, 3, MPI_INT, st.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD); //Отправка индексов и пустоты блока
                        if (hdr[2]) {
                            send_block(m->blocks[block_index], st.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD);
                        }
                        tasks_in_proceed++;
                    }
                    else { //Пока воркер должен стоять в простое (вызвать барьер) - это должны получить все воркеры
                        int dummy = 1;
                        MPI_Send(&dummy, 1, MPI_INT, st.MPI_SOURCE, TAG_WAIT, MPI_COMM_WORLD);

                        waiting_workers.push_back(st.MPI_SOURCE);
                    }
                }
                else {
                    if (st.MPI_TAG == TAG_GET_DATA) { // Это значит, что он хочет получить новые блоки
                        int hdr[3]; // Столбец, две строки в порядке возрастания (две строки - индексы столбца и строки раскладываемого данным воркером блока)
                        MPI_Recv(hdr, 3, MPI_INT, st.MPI_SOURCE, TAG_GET_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        if (m->blocks[hdr[1] + hdr[0] * nb] == nullptr || m->blocks[hdr[2] + hdr[0] * nb] == nullptr) { //Какой-то из блоков пустой - значит, надо пройти дальше
                            int new_col = hdr[0];
                            while (new_col < hdr[1]) {
                                if (m->blocks[hdr[1] + new_col * nb] && m->blocks[hdr[2] + new_col * nb]) {
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
                                send_block(m->blocks[hdr[1] + new_col * nb], st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                                send_block(m->blocks[hdr[2] + new_col * nb], st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                                send_diag(m->diagonals[new_col], st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                            }
                        }
                        else {
                            send_block(m->blocks[hdr[1] + hdr[0] * nb], st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                            send_block(m->blocks[hdr[2] + hdr[0] * nb], st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                            send_diag(m->diagonals[hdr[0]], st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                        }
                    }
                    else {
                        if (st.MPI_TAG == TAG_RESULT) { // Это значит, что он досчитал блок
                            int hdr[3]; //индексы блока и заполненность
                            MPI_Recv(hdr, 3, MPI_INT, st.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            if (hdr[2]) {
                                if (!(m->blocks[hdr[0] + hdr[1] * nb])) m->blocks[hdr[0] + hdr[1] * nb] = new block;
                                recv_block(m->blocks[hdr[0] + hdr[1] * nb], st.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD);
                            }
                            else {
                                if ((m->blocks[hdr[0] + hdr[1] * nb])) {
                                    delete m->blocks[hdr[0] + hdr[1] * nb];
                                    m->blocks[hdr[0] + hdr[1] * nb] = nullptr;
                                }
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
            for (auto i : waiting_workers) {
                int dummy = 1;
                MPI_Send(&dummy, 1, MPI_INT, i, TAG_START, MPI_COMM_WORLD);
            }
            waiting_workers.clear();
        }
        int dummy = 1;
        for (int i = 1; i < size; ++i) {
            MPI_Recv(&dummy, 1, MPI_INT, i, TAG_READY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&dummy, 1, MPI_INT, i, TAG_STOP, MPI_COMM_WORLD);
        }
        end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Computation LDLT time: " << duration.count() << " milliseconds" << std::endl;
        std::cout << "Computation LDLT time: " << duration.count() / 1000.0 << " seconds" << std::endl;

        vect** B = random_vector_generation(m->size);
        std::ofstream out("b.txt");
        for (int i = 0; i < m->size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                out << B[i]->values[j] << std::endl;
            }
        }
        out.close();
        write_decomp_to_file(m, "LDLT_decomp.txt");
        std::cout << "Starting solving SLAE" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
        //Решение нижнетреугольной СЛАУ
        std::cout << "Solving lower_triag SLAE" << std::endl;
        for (int column_n = 0; column_n < nb; ++column_n) {
            solve_L_SLAE(m->blocks[column_n * nb + column_n], B[column_n]);
            int last_index = column_n;
            int tasks_in_proceed = 0;
            int dummy;
            std::vector<int> waiting_workers;
            while (last_index < nb || tasks_in_proceed) {
                MPI_Status st;
                MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
                
                switch (st.MPI_TAG) {
                case TAG_READY: {
                    MPI_Recv(&dummy, 1, MPI_INT, st.MPI_SOURCE, TAG_READY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    int col = last_index + 1;
                    while (col < nb) {
                        if (m->blocks[col + column_n * nb]) {
                            break;
                        }
                        col++;
                    }
                    last_index = col;
                    if (last_index >= nb) {
                        MPI_Send(&dummy, 1, MPI_INT, st.MPI_SOURCE, TAG_WAIT, MPI_COMM_WORLD);
                        waiting_workers.push_back(st.MPI_SOURCE);
                    }
                    else {
                        MPI_Send(&last_index, 1, MPI_INT, st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                        send_block(m->blocks[last_index + column_n * nb], st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                        send_vect(B[column_n], st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                        send_vect(B[last_index], st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                        tasks_in_proceed++;
                    }
                    break;
                }
                case TAG_RESULT: {
                    int row;
                    MPI_Recv(&row, 1, MPI_INT, st.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    recv_vect(B[row], st.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD);
                    tasks_in_proceed--;
                    break;
                }
                default:
                    break;
                }
            }
            for (auto i : waiting_workers) {
                int dummy = 1;
                MPI_Send(&dummy, 1, MPI_INT, i, TAG_START, MPI_COMM_WORLD);
            }
            waiting_workers.clear();
        }
        dummy = 1;
        for (int i = 1; i < size; ++i) {
            MPI_Recv(&dummy, 1, MPI_INT, i, TAG_READY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&dummy, 1, MPI_INT, i, TAG_STOP, MPI_COMM_WORLD);
        }
        //Решение диагональной СЛАУ
        std::cout << "Solving diagonal SLAE" << std::endl;
        int column_n = 0;
        int tasks_in_proceed = 0;
        std::vector<int> waiting_workers;
        while (column_n < nb || tasks_in_proceed) {
            MPI_Status st;
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
            int dummy;
            switch (st.MPI_TAG) {
            case TAG_READY:
                MPI_Recv(&dummy, 1, MPI_INT, st.MPI_SOURCE, TAG_READY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (column_n == nb) {
                    MPI_Send(&dummy, 1, MPI_INT, st.MPI_SOURCE, TAG_WAIT, MPI_COMM_WORLD);
                    waiting_workers.push_back(st.MPI_SOURCE);
                }
                else {
                    MPI_Send(&column_n, 1, MPI_INT, st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                    send_diag(m->diagonals[column_n], st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                    send_vect(B[column_n], st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                    tasks_in_proceed++;
                    column_n++;
                }
                break;
            case TAG_RESULT: {
                int row;
                MPI_Recv(&row, 1, MPI_INT, st.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                recv_vect(B[row], st.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD);
                tasks_in_proceed--;
                break;
            }
            default:
                break;
            }
        }
        for (auto i : waiting_workers) {
            int dummy = 1;
            MPI_Send(&dummy, 1, MPI_INT, i, TAG_START, MPI_COMM_WORLD);
        }
        waiting_workers.clear();
        dummy = 1;
        for (int i = 1; i < size; ++i) {
            MPI_Recv(&dummy, 1, MPI_INT, i, TAG_READY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&dummy, 1, MPI_INT, i, TAG_STOP, MPI_COMM_WORLD);
        }
        //Решение верхнетреугольной СЛАУ
        std::cout << "Solving upper_triag SLAE" << std::endl;
        for (int column_n = nb - 1; column_n >= 0; --column_n) {
            solve_LT_SLAE(m->blocks[column_n * nb + column_n], B[column_n]);
            int last_index = column_n;
            int tasks_in_proceed = 0;
            int dummy;
            std::vector<int> waiting_workers;
            while (last_index >= 0 || tasks_in_proceed) {
                MPI_Status st;
                MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st);

                switch (st.MPI_TAG) {
                case TAG_READY: {
                    MPI_Recv(&dummy, 1, MPI_INT, st.MPI_SOURCE, TAG_READY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    int col = last_index - 1;
                    while (col >= 0) {
                        if (m->blocks[col * nb + column_n]) {
                            break;
                        }
                        col--;
                    }
                    last_index = col;
                    if (last_index <= -1) {
                        MPI_Send(&dummy, 1, MPI_INT, st.MPI_SOURCE, TAG_WAIT, MPI_COMM_WORLD);
                        waiting_workers.push_back(st.MPI_SOURCE);
                    }
                    else {
                        MPI_Send(&last_index, 1, MPI_INT, st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                        send_block(m->blocks[last_index * nb + column_n], st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                        send_vect(B[column_n], st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                        send_vect(B[last_index], st.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
                        tasks_in_proceed++;
                    }
                    break;
                }
                case TAG_RESULT: {
                    int row;
                    MPI_Recv(&row, 1, MPI_INT, st.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    recv_vect(B[row], st.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD);
                    tasks_in_proceed--;
                    break;
                }
                default:
                    break;
                }
            }
            for (auto i : waiting_workers) {
                int dummy = 1;
                MPI_Send(&dummy, 1, MPI_INT, i, TAG_START, MPI_COMM_WORLD);
            }
            waiting_workers.clear();
        }
        dummy = 1;
        for (int i = 1; i < size; ++i) {
            MPI_Recv(&dummy, 1, MPI_INT, i, TAG_READY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&dummy, 1, MPI_INT, i, TAG_STOP, MPI_COMM_WORLD);
        }
        end_time = std::chrono::high_resolution_clock::now();
        //Вывод результата
        std::ofstream out_res("res.txt");
        out_res << std::scientific << std::setprecision(19);
        for (int i = 0; i < nb; ++i) {
            for (int j = 0; j < block_size; ++j) {
                out_res << B[i]->values[j] << std::endl;
            }
        }
        out_res.close();
        std::ofstream out_diag("diag.txt");
        out_diag << std::scientific << std::setprecision(19);
        for (int i = 0; i < nb; ++i) {
            for (int j = 0; j < block_size; ++j) {
                out_diag << m->diagonals[i]->values[j] << std::endl;
            }
        }
        out_diag.close();
        auto duration_2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Computation SLAE time: " << duration_2.count() << " milliseconds" << std::endl;
        std::cout << "Computation SLAE time: " << duration_2.count() / 1000.0 << " seconds" << std::endl;
        std::cout << "SLAE is solved, solution is in file res.txt" << std::endl;
        double solution_acc = check_solution(convert_to_forward_slashes(argv[1]), "res.txt", "b.txt");
        std::cout << "Accuracy = " << solution_acc << std::endl;
        if (m != nullptr) {
            // Очистка блоков
            for (int i = 0; i < nb * nb; ++i) {
                if (m->blocks[i] != nullptr) {
                    delete m->blocks[i];
                }
            }
            delete[] m->blocks;

            // Очистка диагоналей
            for (int i = 0; i < nb; ++i) {
                if (m->diagonals[i] != nullptr) {
                    delete m->diagonals[i];
                }
            }
            delete[] m->diagonals;

            // Очистка самой структуры matrix
            delete m;
            m = nullptr;

            std::cout << "Matrix memory cleaned up" << std::endl;
        }
    }
    else { //Workers
        int dummy = 1;
        block* upper= new block;
        block* diag_b = new block;
        block* lower = new block;
        block* work = new block;
        diagonal* diag = new diagonal;
        bool keep_going = true;
        while (keep_going) {
            MPI_Send(&dummy, 1, MPI_INT, 0, TAG_READY, MPI_COMM_WORLD);
            MPI_Status st;
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
            if (st.MPI_TAG == TAG_WAIT) {
                MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_WAIT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_START, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
                        case TAG_DIAG: {
                            recv_block(diag_b, 0, TAG_DIAG, MPI_COMM_WORLD);
                            recv_diag(diag, 0, TAG_DIAG, MPI_COMM_WORLD);
                            col = b_col;
                            auto is_there_nan = calc_block_final_non_parallel(work, diag_b, diag);
                            if (is_there_nan == -1) {
                                std::cerr << "Nan occured in block " << b_row <<", " << b_col << std::endl;
                                MPI_Abort(MPI_COMM_WORLD, 1);
                            }
                            break;
                        }
                        case TAG_NEW_COLUMN:
                            MPI_Recv(&col, 1, MPI_INT, 0, TAG_NEW_COLUMN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            recv_block(upper, 0, TAG_DATA, MPI_COMM_WORLD);
                            recv_block(lower, 0, TAG_DATA, MPI_COMM_WORLD);
                            recv_diag(diag, 0, TAG_DATA, MPI_COMM_WORLD);
                            calc_block_non_parallel(work, upper, lower, diag);
                            break;
                        case TAG_DATA:
                            recv_block(upper, 0, TAG_DATA, MPI_COMM_WORLD);
                            recv_block(lower, 0, TAG_DATA, MPI_COMM_WORLD);
                            recv_diag(diag, 0, TAG_DATA, MPI_COMM_WORLD);
                            calc_block_non_parallel(work, upper, lower, diag);
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
                        hdr[2] = calc_block_final_non_parallel(work, diag_b, diag);
                        if (hdr[2] == -1) {
                            std::cerr << "Nan occured in block " << b_row << ", " << b_col << std::endl;
                            MPI_Abort(MPI_COMM_WORLD, 1);
                        }
                    }
                    MPI_Send(&hdr, 3, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
                    if (hdr[2]) {
                        send_block(work, 0, TAG_RESULT, MPI_COMM_WORLD);
                    }
                }
                else {
                    if (st.MPI_TAG == TAG_STOP) {
                        MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_STOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        keep_going = false;
                    }
                }
            }
        }
        delete upper;
        delete lower;
        delete work;
        delete diag_b;
        delete diag;
        //Решение нижнетреугольной СЛАУ
        work = new block;
        vect* x = new vect;
        vect* b = new vect;
        keep_going = true;
        while (keep_going) {
            MPI_Send(&dummy, 1, MPI_INT, 0, TAG_READY, MPI_COMM_WORLD);
            MPI_Status st;
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
            switch (st.MPI_TAG) {
            case TAG_WAIT:
                MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_WAIT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_START, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                break;
            case TAG_DATA:
                MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                recv_block(work, 0, TAG_DATA, MPI_COMM_WORLD);
                recv_vect(x, 0, TAG_DATA, MPI_COMM_WORLD);
                recv_vect(b, 0, TAG_DATA, MPI_COMM_WORLD);
                sub_block_mul_sol(work, x, b);
                MPI_Send(&dummy, 1, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
                send_vect(b, 0, TAG_RESULT, MPI_COMM_WORLD);
                break;
            case TAG_STOP:
                MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_STOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                keep_going = false;
                break;
            default:
                break;
            }
        }
        delete work;
        delete x;
        delete b;

        //Решение диагональной СЛАУ
        diagonal* d = new diagonal;
        b = new vect;
        keep_going = true;
        while (keep_going) {
            MPI_Send(&dummy, 1, MPI_INT, 0, TAG_READY, MPI_COMM_WORLD);
            MPI_Status st;
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
            switch (st.MPI_TAG) {
            case TAG_WAIT:
                MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_WAIT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_START, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                break;
            case TAG_DATA:
                MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                recv_diag(d, 0, TAG_DATA, MPI_COMM_WORLD);
                recv_vect(b, 0, TAG_DATA, MPI_COMM_WORLD);
                solve_D_SLAE(d, b);
                MPI_Send(&dummy, 1, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
                send_vect(b, 0, TAG_RESULT, MPI_COMM_WORLD);
                break;
            case TAG_STOP:
                MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_STOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                keep_going = false;
                break;
            default:
                break;
            }
        }
        delete d;
        delete b;

        //Решение верхнетреугольной СЛАУ
        work = new block;
        x = new vect;
        b = new vect;
        keep_going = true;
        while (keep_going) {
            MPI_Send(&dummy, 1, MPI_INT, 0, TAG_READY, MPI_COMM_WORLD);
            MPI_Status st;
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
            switch (st.MPI_TAG) {
            case TAG_WAIT:
                MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_WAIT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_START, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                break;
            case TAG_DATA:
                MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                recv_block(work, 0, TAG_DATA, MPI_COMM_WORLD);
                recv_vect(x, 0, TAG_DATA, MPI_COMM_WORLD);
                recv_vect(b, 0, TAG_DATA, MPI_COMM_WORLD);
                sub_block_T_mul_sol(work, x, b);
                MPI_Send(&dummy, 1, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
                send_vect(b, 0, TAG_RESULT, MPI_COMM_WORLD);
                break;
            case TAG_STOP:
                MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_STOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                keep_going = false;
                break;
            default:
                break;
            }
        }
        delete work;
        delete x;
        delete b;
    }
    MPI_Type_free(&MPI_BLOCK);
    MPI_Type_free(&MPI_DIAG);
    std::cout << rank << std::endl;
    MPI_Finalize();
}
