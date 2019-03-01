#define BENCHMARK "OSU MPI%s All-to-All Personalized Exchange Latency Test"
/*
 * Copyright (C) 2002-2018 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 * Copyright (C) 2019 Institute of Information Technology, Mathematics
 * and Mechanics (IITMM), University of Nizhny Novgorod.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#include <osu_util.h>

int main (int argc, char *argv[])
{
    int myid, numprocs;

    set_header(HEADER);
    set_benchmark_name("osu_latency");

    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));

    MPI_CHECK(MPI_Finalize());

    return EXIT_SUCCESS;
}
