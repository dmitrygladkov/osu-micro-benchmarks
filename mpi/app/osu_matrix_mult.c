#define BENCHMARK "OSU MPI%s Matrix Multiplication Test"
/*
 * Copyright (C) 2002-2019 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 * Copyright (C) 2019 Institute of Information Technology, Mathematics
 * and Mechanics (IITMM), University of Nizhny Novgorod.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *          Dmitry Gladkov (dgladkov1709@gmail.com)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#include <osu_util.h>

#define MASTER_2_WORKER 100
#define WORKER_2_MASTER 200

static void dummy_data_init(double **matrixA, double **matrixB, size_t size)
{
    size_t i, j;

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            matrixA[i][j] = i + j;
            matrixB[j][i] = i * j;
        }
    }
}

static void matrix_mult(double **matrixA, double **matrixB,
                        double **res_matrix, int size, int rows, int block)
{
    int i, j, k;

    for (j = 0; j < rows; j++) {
        for (k = 0; k < size; k++) {
            for (i = 0; i < size; i++) {
                res_matrix[j][i] += matrixA[j][k] * matrixB[k][i];
            }
        }
    }
}

static double *attach_mem_block(void *memblock, size_t offset, size_t size)
{
    if (!memblock) {
        void *new = calloc(1, size);
        return new;
    }

    return memblock + offset;
}

static void dettach_mem_block(void *memblock, double *assigned_to)
{
    if (!memblock) {
        free(assigned_to);
    }
}

static void print_matrix(const char *name, double **matrix, size_t size, int should_print)
{
    size_t i, j;

    if (should_print) {
        printf("******************************************************\n");
        printf("Matrix %s:\n", name);
        for (i = 0; i < size; i++) {
            printf("\n");
            for (j = 0; j < size; j++) 
                printf("%6.2f   ", matrix[i][j]);
        }
        printf("\n******************************************************\n");
    }
}

static size_t matrix_offset = 0;

static void generate_test_matrix_data(int myid, void *memblocks[3], size_t memblock_max_size,
                                      double **matrixA, double **matrixB, double **res_matrix,
                                      size_t size, size_t double_size, int should_print)
{
    size_t i;

    if (((matrix_offset + 1) * size * double_size * size) > memblock_max_size) {
        matrix_offset = 0;
    }

    for (i = 0; i < size; i++) {
        matrixA[i] = attach_mem_block(memblocks[0],
                                      i * size * double_size +
                                      (matrix_offset * size * double_size * size),
                                      size * double_size);
        memset(matrixA[i], 0, size * double_size);

        matrixB[i] = attach_mem_block(memblocks[1],
                                      i * size * double_size +
                                      (matrix_offset * size * double_size * size),
                                      size * double_size);
        memset(matrixB[i], 0, size * double_size);

        res_matrix[i] = attach_mem_block(memblocks[2],
                                         i * size * double_size +
                                         (matrix_offset * size * double_size * size),
                                         size * double_size);
        memset(res_matrix[i], 0, size * double_size);
    }

    if (myid == 0) {
        dummy_data_init(matrixA, matrixB, size);

        print_matrix("A", matrixA, size, should_print);
        print_matrix("B", matrixB, size, should_print);
    }

    matrix_offset++;
}

static void free_test_matrix_data(void *memblocks[3], double **matrixA,
                                  double **matrixB, double **res_matrix, size_t size)
{
    size_t i;

    for (i = 0; i < size; i++) {
        dettach_mem_block(memblocks[0], matrixA[i]);
        dettach_mem_block(memblocks[1], matrixB[i]);
        dettach_mem_block(memblocks[2], res_matrix[i]);
    }
}

int main(int argc, char *argv[])
{
    int myid, numprocs, dest;
    int po_ret;
    double **matrixA, **matrixB, **res_matrix;
    void *memblocks[3] = {};
    int size, rows, double_size, exit_status = EXIT_SUCCESS;
    size_t iter;
    double t_start = 0.0, t_end = 0.0, total_time = 0.0;
    double compute_time = 0.0, recv_time = 0.0;
    int i, full, extra, offset, mtype;
    MPI_Status status;
    int reuse_memory = 1, print_matrices = 0;
    char *p;
    size_t memblock_max_size;

    options.bench = APP;
    options.subtype = LAT;

    set_header(HEADER);
    set_benchmark_name("osu_matrix_mult");

    po_ret = process_options(argc, argv);
    if (PO_OKAY == po_ret && NONE != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));
    
    if (0 == myid) {
        switch (po_ret) {
            case PO_CUDA_NOT_AVAIL:
                fprintf(stderr, "CUDA support not enabled.  Please recompile "
                        "benchmark with CUDA support.\n");
                break;
            case PO_OPENACC_NOT_AVAIL:
                fprintf(stderr, "OPENACC support not enabled.  Please "
                        "recompile benchmark with OPENACC support.\n");
                break;
            case PO_BAD_USAGE:
                print_bad_usage_message(myid);
                break;
            case PO_HELP_MESSAGE:
                print_help_message(myid);
                break;
            case PO_VERSION_MESSAGE:
                print_version_message(myid);
                MPI_CHECK(MPI_Finalize());
                exit(EXIT_SUCCESS);
            case PO_OKAY:
                break;
        }
    }

    switch (po_ret) {
        case PO_CUDA_NOT_AVAIL:
        case PO_OPENACC_NOT_AVAIL:
        case PO_BAD_USAGE:
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_FAILURE);
        case PO_HELP_MESSAGE:
        case PO_VERSION_MESSAGE:
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_SUCCESS);
        case PO_OKAY:
            break;
    }

    MPI_CHECK(MPI_Type_size(MPI_DOUBLE, &double_size));

    memblock_max_size = options.max_message_size * options.max_message_size * double_size;

    matrixA = calloc(1, options.max_message_size * sizeof(*matrixA));
    matrixB = calloc(1, options.max_message_size * sizeof(*matrixB));
    res_matrix = calloc(1, options.max_message_size * sizeof(*res_matrix));

    p = getenv("OSU_APP_REUSE_MEMORY");
    if (p != NULL) {
        reuse_memory = atoi(p);
    }

    p = getenv("OSU_APP_PRINT_MATRIX");
    if (p != NULL) {
        print_matrices = atoi(p);
    }

    print_header(myid, LAT);

    for (size = options.min_message_size; size <= options.max_message_size; size = (size ? size * 2 : 1)) {
        if (size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }

        if (size >= 64) {
            if (size >= 512) {
                options.skip = 1;
                //options.iterations = 1;
            } else {
                options.skip /= 2;
                //options.iterations /= 2;

                if (options.skip == 0) {
                    options.skip = 1;
                }

                if (options.iterations == 0) {
                    options.iterations = 1;
                }
            }
        }

        if (numprocs == 1) {
            for (iter = 0; iter < options.iterations + options.skip; iter++) {
                if (!reuse_memory || (iter == 0)) {
                    memblocks[0] = calloc(1, memblock_max_size);
                    memblocks[1] = calloc(1, memblock_max_size);
                    memblocks[2] = calloc(1, memblock_max_size);
                }
                generate_test_matrix_data(myid, memblocks, memblock_max_size,
                                          matrixA, matrixB, res_matrix, size,
                                          double_size, print_matrices);

                if (iter >= options.skip) {
                    t_start = MPI_Wtime();
                }

                matrix_mult(matrixA, matrixB, res_matrix, size, size, 0);

                if (iter >= options.skip) {
                    t_end = MPI_Wtime();

                    total_time += (t_end - t_start);
                }

                print_matrix("result", res_matrix, size, print_matrices);

                free_test_matrix_data(memblocks, matrixA, matrixB, res_matrix, size);
                if (!reuse_memory || (iter == (options.iterations + options.skip - 1))) {
                    free(memblocks[0]);
                    free(memblocks[1]);
                    free(memblocks[2]);
                }
            }

            compute_time = total_time;

            goto res;
        } /* no code should be executed after this statement in case of numprocs=1 */

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        if (myid == 0) {
            for(iter = 0; iter < options.iterations + options.skip; iter++) {
                if (!reuse_memory || (iter == 0)) {
                    memblocks[0] = calloc(1, memblock_max_size);
                    memblocks[1] = calloc(1, memblock_max_size);
                    memblocks[2] = calloc(1, memblock_max_size);
                }
                generate_test_matrix_data(myid, memblocks, memblock_max_size,
                                          matrixA, matrixB, res_matrix, size,
                                          double_size, print_matrices);
                MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

                if (iter >= options.skip) {
                    t_start = MPI_Wtime();
                }

                /* Send matrix data to the worker tasks */
                full = size / (numprocs - 1);
                extra = size % (numprocs - 1);
                offset = 0;

                mtype = MASTER_2_WORKER;

                for (dest = 1; dest < numprocs; dest++) {
                    rows = (dest <= extra) ? full + 1 : full;

                    //printf("Sending %d rows to task %d offset=%d\n", rows, dest, offset);

                    MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
                    MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
                    MPI_Send(&(matrixA[offset][0]), rows * size, MPI_DOUBLE, dest, mtype,
                             MPI_COMM_WORLD);
                    MPI_Send(&(matrixB[0][0]), size * size, MPI_DOUBLE, dest, mtype,
                             MPI_COMM_WORLD);
                    offset = offset + rows;
                }

                /* Receive results from worker tasks */
                mtype = WORKER_2_MASTER;
                for (i = 1; i < numprocs; i++) {
                    MPI_Recv(&offset, 1, MPI_INT, i, mtype, MPI_COMM_WORLD, &status);
                    MPI_Recv(&rows, 1, MPI_INT, i, mtype, MPI_COMM_WORLD, &status);
                    MPI_Recv(&(res_matrix[offset][0]), rows * size, MPI_DOUBLE, i, mtype, 
                             MPI_COMM_WORLD, &status);
                }
   
                if (iter >= options.skip) {
                    t_end = MPI_Wtime();

                    total_time += (t_end - t_start);
                }

                print_matrix("result", res_matrix, size, print_matrices);

                free_test_matrix_data(memblocks, matrixA, matrixB, res_matrix, size);
                if (!reuse_memory || (iter == (options.iterations + options.skip - 1))) {
                    free(memblocks[0]);
                    free(memblocks[1]);
                    free(memblocks[2]);
                }
            }

            MPI_Reduce(&compute_time, &recv_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            compute_time = recv_time;
        } else {
            for(iter = 0; iter < options.iterations + options.skip; iter++) {
                if (!reuse_memory || (iter == 0)) {
                    memblocks[0] = calloc(1, memblock_max_size);
                    memblocks[1] = calloc(1, memblock_max_size);
                    memblocks[2] = calloc(1, memblock_max_size);
                }
                generate_test_matrix_data(myid, memblocks, memblock_max_size,
                                          matrixA, matrixB, res_matrix,
                                          size, double_size, 0);
                MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

                mtype = MASTER_2_WORKER;

                MPI_Recv(&offset, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
                MPI_Recv(&rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
                MPI_Recv(&(matrixA[0][0]), rows * size, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
                MPI_Recv(&(matrixB[0][0]), size * size, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);

                //printf("Received %d rows to task %d offset=%d\n", rows, myid, offset);
                t_start = MPI_Wtime();
                matrix_mult(matrixA, matrixB, res_matrix, size, rows, 0);
                t_end = MPI_Wtime();

                mtype = WORKER_2_MASTER;

                MPI_Send(&offset, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
                MPI_Send(&rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
                MPI_Send(&(res_matrix[0][0]), rows * size, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);

                free_test_matrix_data(memblocks, matrixA, matrixB, res_matrix, size);
                if (!reuse_memory || (iter == (options.iterations + options.skip - 1))) {
                    free(memblocks[0]);
                    free(memblocks[1]);
                    free(memblocks[2]);
                }

                compute_time += t_end - t_start;
            }

            MPI_Reduce(&compute_time, &recv_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }

res:
        if (myid == 0) {
            double time = (total_time) * 1e6 / (2.0 * options.iterations);
            double comp = (compute_time) * 1e6 / (2.0 * options.iterations);

            fprintf(stdout, "%-*d%*.*f%*.*f\n", 10, size,
                    FIELD_WIDTH, FLOAT_PRECISION, comp,
                    FIELD_WIDTH, FLOAT_PRECISION, time);
            fflush(stdout);
        }
    }

    free(matrixA);
    free(matrixB);
    free(res_matrix);

out:
    MPI_CHECK(MPI_Finalize());
    return exit_status;
}
