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
        std::cout << "\nexchange-mpi" << std::endl;
    }

    for (int count = 2; count <= 4; count *= 2)
    {
        char sendbuff[count];
        char recvbuff[count];
        std::fill_n(sendbuff, count, rank);

        int loopCount = std::ceil(std::log2(world_size));

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        int distance = 1;
        for (int i = 0; i < loopCount; i++)
        {
            int src = (rank + distance) % world_size;
            int dst = (rank - distance + world_size) % world_size;

            std::cout << "Count " << count << ": Rank " << rank << " sending to rank " << dst << std::endl;

            MPICHECK(MPI_Sendrecv(sendbuff, count, MPI_CHAR, src, 0, recvbuff, count, MPI_CHAR, dst, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

            distance *= 2;
        }

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        print_buffer(rank, "recvbuff", recvbuff, count, count);
    }

    MPICHECK(MPI_Finalize());
    return 0;
}