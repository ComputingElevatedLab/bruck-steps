#include <iostream>
#include <iomanip>

#include "mpi.h"

#include "../../common/error-catch.h"

void print_buffer(int rank, std::string name, int *buffer, int count)
{
    std::cout << std::fixed << "Count " << count << ": Rank " << rank << " " << name << " buffer: [";
    for (int i = 0; i < count; i++)
    {
        std::cout << " " << buffer[i] << " ";
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
        std::cout << "\npingpong-mpi" << std::endl;
    }

    for (int count = 2; count <= 4; count *= 2)
    {
        int sendbuff[count];
        int recvbuff[count];
        std::fill_n(sendbuff, count, rank);
        std::fill_n(recvbuff, count, -1);

        int src = 0;
        int dst = world_size - 1;

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        if (rank == src)
        {
            print_buffer(rank, "sendbuff", sendbuff, count);
        }

        if (rank == dst)
        {
            std::cout << "Count " << count << ": Rank " << rank << " receiving from rank 0" << std::endl;
            MPICHECK(MPI_Recv(&recvbuff, count, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        }
        else if (rank == src)
        {
            std::cout << "Count " << count << ": Rank " << rank << " sending to rank " << dst << std::endl;
            MPICHECK(MPI_Send(&sendbuff, count, MPI_INT, dst, 0, MPI_COMM_WORLD));
        }

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        if (rank == dst)
        {
            print_buffer(rank, "recvbuff", recvbuff, count);
        }
    }

    MPICHECK(MPI_Finalize());
    return 0;
}