#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>

#include "mpi.h"

#include "../../common/error-catch.h"
#include "../../common/utils.h"

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
        std::cout << "\nbruck-mpi" << std::endl;
    }

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    int radix = 2;

    for (int count = 2; count <= 4; count *= 2)
    {
        int bytes = count * world_size;
        char sendbuff[bytes];
        char recvbuff[bytes];
        std::fill_n(sendbuff, bytes, rank);

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        print_buffer(rank, "sendbuff", sendbuff, count, count);

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        /* Bruck Start */
        int w = std::ceil(std::log(world_size) / std::log(radix));
        int nlpow = myPow(radix, w - 1);
        int d = (myPow(radix, w) - world_size) / nlpow;

        std::memcpy(recvbuff, sendbuff, world_size * count);
        std::memcpy(&sendbuff[(world_size - rank) * count], recvbuff, rank * count);
        std::memcpy(sendbuff, &recvbuff[rank * count], (world_size - rank) * count);

        int *rank_r_reps = new int[world_size * w * sizeof(int)];
        for (int i = 0; i < world_size; i++)
        {
            std::vector<int> r_rep = convert10tob(w, i, radix);
            std::memcpy(&rank_r_reps[i * w], r_rep.data(), w * sizeof(int));
        }

        int sent_blocks[nlpow];
        int di = 0;
        int ci = 0;

        char *tempbuff = new char[nlpow * count];

        for (int x = 0; x < w; x++)
        {
            int ze = (x == w - 1) ? radix - d : radix;
            for (int z = 1; z < ze; z++)
            {
                di = 0;
                ci = 0;
                for (int i = 0; i < world_size; i++)
                {
                    if (rank_r_reps[i * w + x] == z)
                    {
                        sent_blocks[di++] = i;
                        memcpy(&tempbuff[count * ci++], &sendbuff[count * i], count);
                    }
                }

                int distance = z * myPow(radix, x);
                int recv_proc = (rank - distance + world_size) % world_size;
                int send_proc = (rank + distance) % world_size;
                long long comm_size = di * count;

                MPI_Sendrecv(tempbuff, comm_size, MPI_CHAR, send_proc, 0, recvbuff, comm_size, MPI_CHAR, recv_proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                for (int i = 0; i < di; i++)
                {
                    long long offset = sent_blocks[i] * count;
                    memcpy(sendbuff + offset, recvbuff + (count * i), count);
                }
            }
        }

        for (int i = 0; i < world_size; i++)
        {
            int index = (rank - i + world_size) % world_size;
            memcpy(&recvbuff[index * count], &sendbuff[count * i], count);
        }
        /* Bruck End */

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        print_buffer(rank, "recvbuff", recvbuff, count, bytes);
    }

    MPICHECK(MPI_Finalize());
    return 0;
}