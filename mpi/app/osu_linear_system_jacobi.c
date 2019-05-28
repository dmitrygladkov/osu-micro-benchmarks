#define BENCHMARK "OSU MPI%s Jacobi method for linear system Test"
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

const double eps = 0.001;

static void dummy_data_init(double *matrixA, double *vectorB,
                            double *vectorX, size_t dim)
{
    size_t i, j;

    srand(13);
    for (i = 0; i < dim; ++i) {
        for (j = 0; j < dim; ++j) {
            if (i != j) {
                matrixA[i * dim + j] = 1;
            } else {
                matrixA[i * dim + j] = dim;
            }
        }

        vectorB[i] = rand() % 10;
        vectorX[i] = 0;
    }
}

static int check_matrix(double *matrixA, size_t dim)
{
    double sum = 0;
    size_t i, j;

    for (i = 0; i < dim; ++i) {
        sum = 0;
        for (j = 0; j < dim; ++j) {
            sum += fabs(matrixA[i * dim + j]);
        }

        sum -= matrixA[i * dim + i];
        if (matrixA[i * dim + i] <= sum) {
            return 0;
        }
    }

    return 1;
}

static inline double get_norm(double* x1, double* x2, size_t dim)
{
    double res = 0;
    double sub = 0;
    size_t i;

    for (i = 0; i < dim; ++i) {
        sub = x1[i] - x2[i];
        res += sub * sub;
    }

    return sqrt(res);
}

static void jacobi(double *matrixA, double *vectorB, double *vectorRes,
                   double *vectorX0, double *matrixB, double *vectorD,
                   double *vectorX, double *vectorOldX,
                   size_t dim, size_t double_size)
{
    double sum;
    size_t i, j;

    for (i = 0; i < dim; ++i) {
        for (j = 0; j < dim; ++j) {
            if (i == j) {
                matrixB[i * dim + j] = 0;
            } else {
                matrixB[i * dim + j] =
                    - matrixA[i * dim + j] / matrixA[i * dim + i];
            }
        }
        vectorD[i] = vectorB[i] / matrixA[i * dim + i];
    }

    memcpy(vectorOldX, vectorX0, dim * double_size);

    double diff = 0.;

    do {
        for(i = 0; i < dim; ++i) {
            sum = vectorD[i];
            for(j = 0; j < dim; ++j) {
                sum += vectorOldX[j] * matrixB[i * dim + j];
            }
            vectorX[i] = sum;
        }

        diff = get_norm(vectorX, vectorOldX, dim);

        memcpy(vectorOldX, vectorX, dim * double_size);
    } while (diff >= eps);

    memcpy(vectorRes, vectorX, dim * double_size);
}

static double p_jacobi(double *matrixA, double *vectorB, double *vectorRes,
                       double *vectorX0, double *matrixB, double *vectorD,
                       double *vectorX, double *vectorOldX, double *vectorLocX,
                       size_t dim, size_t num_of_row, size_t double_size, int myid)
{
    double comp_time = 0, t_start, t_end;
    size_t i, j;

    if (myid == 0) {
        MPI_Scatter(matrixA, num_of_row * dim, MPI_DOUBLE, /*matrixA*/MPI_IN_PLACE,
                    num_of_row * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(vectorB, num_of_row, MPI_DOUBLE, /*vectorB*/MPI_IN_PLACE,
                    num_of_row, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatter(matrixA, num_of_row * dim, MPI_DOUBLE, matrixA,
                    num_of_row * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(vectorB, num_of_row, MPI_DOUBLE, vectorB,
                    num_of_row, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    t_start = MPI_Wtime();
    {
        for (i = 0; i < num_of_row; ++i) {
            for (j = 0; j < dim; ++j) {
                if (i + myid * num_of_row == j) {
                    matrixB[i * dim + j] = 0;
                } else {
                    matrixB[i * dim + j] =
                        - matrixA[i * dim + j] / matrixA[i * dim + i +
                                                         myid * num_of_row];
                }
            }
            vectorD[i] = vectorB[i] / matrixA[i * dim + i + myid * num_of_row];
        }
    }
    t_end = MPI_Wtime();
    comp_time += (t_end - t_start);

    double diff = 0;
    double *row = NULL;
    double sum;

    do {
        MPI_Bcast(vectorOldX, dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        t_start = MPI_Wtime();
        {
            for (i = 0; i < num_of_row; ++i) {
                sum = vectorD[i];
                row = matrixB + i * dim;
                for(j = 0; j < dim; ++j) {
                    sum += vectorOldX[j] * row[j];
                }
                vectorLocX[i] = sum;
            }
        }
        t_end = MPI_Wtime();
        comp_time += (t_end - t_start);

        MPI_Gather(vectorLocX, num_of_row, MPI_DOUBLE, vectorX,
                   num_of_row, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (myid == 0) {
            t_start = MPI_Wtime();
            diff = get_norm(vectorOldX, vectorX, dim);
            {
                memcpy(vectorOldX, vectorX, dim * double_size);
                t_end = MPI_Wtime();
            }
            comp_time += (t_end - t_start);
        }

        MPI_Bcast(&diff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    } while(diff >= eps);

    return comp_time;
}

int main(int argc, char *argv[])
{
    int myid, numprocs, dest;
    int po_ret;
    double **matrixA, **matrixB, **res_matrix;
    int exit_status = EXIT_SUCCESS;
    double t_start = 0.0, t_end = 0.0, total_time = 0.0;
    double compute_time = 0.0, recv_time = 0.0;
    MPI_Status status;
    int iter, size, double_size;

    options.bench = APP;
    options.subtype = LAT;

    set_header(HEADER);
    set_benchmark_name("osu_linear_system_jacobi");

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
                double *a            = calloc(1, size * size * double_size);
                double *b            = calloc(1, size * double_size);
                double *x0           = calloc(1, size * double_size);
                double *res          = calloc(1, size * double_size);
                double *x_vector     = calloc(1, size * double_size);
                double *b_matrix     = calloc(1, size * size * double_size);;
                double *d_vector     = calloc(1, size * double_size);
                double *x_old_vector = calloc(1, size * double_size);

                dummy_data_init(a, b, x0, size);

                if (!check_matrix(a, size)) {
                    continue;
                }

                if (iter >= options.skip) {
                    t_start = MPI_Wtime();
                }

                jacobi(a, b, res, x0, b_matrix, d_vector,
                       x_vector, x_old_vector,
                       size, double_size);

                if (iter >= options.skip) {
                    t_end = MPI_Wtime();

                    total_time += (t_end - t_start);
                }

                free(a);
                free(b);
                free(x0);
                free(res);
                free(x_vector);
                free(b_matrix);
                free(d_vector);
                free(x_old_vector);
            }

            compute_time = total_time;

            goto res;
        } /* no code should be executed after this statement in case of numprocs=1 */

        size_t num_of_row = size / numprocs;

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        if (myid == 0) {
            for(iter = 0; iter < options.iterations + options.skip; iter++) {
                double *a            = calloc(1, size * size * double_size);
                double *b            = calloc(1, size * double_size);
                double *x0           = calloc(1, size * double_size);
                double *res          = calloc(1, size * double_size);
                double *x_vector     = calloc(1, size * double_size);
                double *b_matrix     = calloc(1, num_of_row * size * double_size);
                double *d_vector     = calloc(1, num_of_row * double_size);
                double *x_old_vector = calloc(1, size * double_size);
                double *x_loc_vector = calloc(1, num_of_row * double_size);

                dummy_data_init(a, b, x0, size);

                int check_res = check_matrix(a, size);

                MPI_Bcast(&check_res, 1, MPI_INT, 0, MPI_COMM_WORLD);
                if (!check_res) {
                    continue;
                }

                memcpy(x_old_vector, x0, size * double_size);

                MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

                if (iter >= options.skip) {
                    t_start = MPI_Wtime();
                }

                double comp_time = p_jacobi(a, b, res, x0, b_matrix, d_vector, x_vector,
                                            x_old_vector, x_loc_vector, size,
                                            num_of_row, double_size, myid);
   
                if (iter >= options.skip) {
                    t_end = MPI_Wtime();
                    total_time   += (t_end - t_start);
                    compute_time += comp_time;
                }

                memcpy(res, x_vector, size * double_size);

                free(a);
                free(b);
                free(x0);
                free(res);
                free(x_vector);
                free(b_matrix);
                free(d_vector);
                free(x_old_vector);
                free(x_loc_vector);
            }

            MPI_Reduce(&compute_time, &recv_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            compute_time += recv_time;
        } else {
            for(iter = 0; iter < options.iterations + options.skip; iter++) {
                /*allocate memory-fill some buffers by NULL*/
                double *a            = calloc(1, size * size * double_size);
                double *b            = calloc(1, size * double_size);
                double *x0           = calloc(1, size * double_size);
                double *res          = calloc(1, size * double_size);
                double *x_vector     = calloc(1, size * double_size);
                double *b_matrix     = calloc(1, num_of_row * size * double_size);
                double *d_vector     = calloc(1, num_of_row * double_size);
                double *x_old_vector = calloc(1, size * double_size);
                double *x_loc_vector = calloc(1, num_of_row * double_size);

                int check_res = 1;
 
                MPI_Bcast(&check_res, 1, MPI_INT, 0, MPI_COMM_WORLD);
                if (!check_res) {
                    continue;
                }
                MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

                
                double comp_time = p_jacobi(a, b, res, x0, b_matrix, d_vector, x_vector,
                                            x_old_vector, x_loc_vector, size,
                                            num_of_row, double_size, myid);

                if (iter >= options.skip) {
                    compute_time += comp_time;
                }

                free(a);
                free(b);
                free(x0);
                free(res);
                free(x_vector);
                free(b_matrix);
                free(d_vector);
                free(x_old_vector);
                free(x_loc_vector);
            }

            MPI_Reduce(&compute_time, &recv_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }

res:
        if (myid == 0) {
            double time     = (total_time) * 1e6 / (2.0 * options.iterations);
            double comp     = (compute_time) * 1e6 / (2.0 * options.iterations);
            double avg_comp = (numprocs == 1) ? comp : (comp / numprocs);

            fprintf(stdout, "%-*d%*.*f%*.*f%*.*f\n", 10, size,
                    FIELD_WIDTH, FLOAT_PRECISION, avg_comp,
                    FIELD_WIDTH, FLOAT_PRECISION, comp,
                    FIELD_WIDTH, FLOAT_PRECISION, time);
            fflush(stdout);
        }
    }

out:
    MPI_CHECK(MPI_Finalize());
    return exit_status;
}
