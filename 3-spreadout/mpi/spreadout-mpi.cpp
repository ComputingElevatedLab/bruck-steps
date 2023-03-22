#include <cmath>
#include <iostream>
#include <iomanip>

#include "mpi.h"

#include "../../common/error-catch.h"

void print_buffer(int rank, std::string name, char *buffer, int count, int size)
{
    std::cout << std::fixed << "Count " << count << ": Rank " << rank << " " << name << " buffer: [";
    for (int i = 0; i < size; i++)
    {
        std::cout << " " << (int)buffer[i] << " ";
    }
    std::cout << "]" << std::endl;
}

int main(int argc, char **argv)
{
    MPICHECK(MPI_Init(&argc, &argv));

    int world_size, rank;
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    if (rank == 0)
    {
        std::cout << "\nspreadout-mpi" << std::endl;
    }

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    for (int count = 2; count <= 4; count *= 2)
    {
        int bytes = count * world_size;
        char sendbuff[bytes];
        char recvbuff[bytes];
        std::fill_n(sendbuff, bytes, rank);

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        print_buffer(rank, "sendbuff", sendbuff, count, count);

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        MPI_Request *req = (MPI_Request *)malloc(2 * world_size * sizeof(MPI_Request));
        MPI_Status *stat = (MPI_Status *)malloc(2 * world_size * sizeof(MPI_Status));
        for (int i = 0; i < world_size; i++)
        {
            int src = (rank + i) % world_size;
            std::cout << "Count " << count << ": Rank " << rank << " receiving from rank " << src << std::endl;
            MPI_Irecv(&recvbuff[src * count], count, MPI_CHAR, src, 0, MPI_COMM_WORLD, &req[i]);
        }

        for (int i = 0; i < world_size; i++)
        {
            int dst = (rank - i + world_size) % world_size;
            std::cout << "Count " << count << ": Rank " << rank << " sending to rank " << dst << std::endl;
            MPI_Isend(&sendbuff[dst * count], count, MPI_CHAR, dst, 0, MPI_COMM_WORLD, &req[i + world_size]);
        }

        MPI_Waitall(2 * world_size, req, stat);

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        print_buffer(rank, "recvbuff", recvbuff, count, bytes);
    }

    MPICHECK(MPI_Finalize());
    return 0;
}