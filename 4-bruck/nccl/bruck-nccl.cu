#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>

#include "mpi.h"
#include "nccl.h"

#include "../../common/error-catch.cu"
#include "../../common/error-catch.h"
#include "../../common/hostname.cu"
#include "../../common/utils.h"

void print_buffer(int rank, std::string name, char *d_buffer, int count, int bytes)
{
    char *h_buffer = new char[count];
    cudaMemcpy(h_buffer, d_buffer, bytes, cudaMemcpyDeviceToHost);
    std::cout << std::fixed << "Count " << count << ": Rank " << rank << " " << name << " buffer: [";
    for (int i = 0; i < bytes; i++)
    {
        std::cout << " " << (int)h_buffer[i] << " ";
    }
    std::cout << "]" << std::endl;
}

int main(int argc, char **argv)
{
    MPICHECK(MPI_Init(&argc, &argv));

    int world_size, rank;
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    /* Begin NCCL Setup Boilerplate*/
    uint64_t hostHashs[world_size];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[rank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));

    int local_rank = 0;
    for (int i = 0; i < world_size; i++)
    {
        if (i == rank)
        {
            break;
        }
        if (hostHashs[i] == hostHashs[rank])
        {
            local_rank++;
        }
    }

    ncclUniqueId id;
    if (rank == 0)
    {
        ncclGetUniqueId(&id);
    }
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    ncclComm_t comm;
    cudaStream_t stream;
    CUDACHECK(cudaSetDevice(local_rank));
    CUDACHECK(cudaStreamCreate(&stream));
    NCCLCHECK(ncclCommInitRank(&comm, world_size, id, rank));
    /* End NCCL Setup Boilerplate*/

    if (rank == 0)
    {
        std::cout << "\nbruck-nccl" << std::endl;
    }

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Host variables
    char *h_sendbuff;
    char *h_recvbuff;
    char *h_verifybuff;

    // Device variables
    char *d_sendbuff;
    char *d_recvbuff;

    int radix = 2;

    for (int count = 10000; count <= 10000; count *= 2)
    {
        int bytes = count * world_size;
        h_sendbuff = new char[bytes];
        h_recvbuff = new char[bytes];
        h_verifybuff = new char[bytes];
        std::fill_n(h_sendbuff, bytes, rank);

        CUDACHECK(cudaMalloc((void **)&d_sendbuff, bytes));
        CUDACHECK(cudaMalloc((void **)&d_recvbuff, bytes));
        CUDACHECK(cudaMemcpy(d_sendbuff, h_sendbuff, bytes, cudaMemcpyHostToDevice));

        // Prepare the verification buffer
        for (int i = 0; i < world_size; i++)
        {
            for (int j = 0; j < count; j++)
            {
                h_verifybuff[j + i * count] = i;
            }
        }

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        // print_buffer(rank, "sendbuff", d_sendbuff, count, count);

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        /* Bruck Start */
        int w = std::ceil(std::log(world_size) / std::log(radix));
        int nlpow = myPow(radix, w - 1);
        int d = (myPow(radix, w) - world_size) / nlpow;

        CUDACHECK(cudaMemcpy(d_recvbuff, d_sendbuff, world_size * count, cudaMemcpyDeviceToDevice));
        CUDACHECK(cudaMemcpy(&d_sendbuff[(world_size - rank) * count], d_recvbuff, rank * count, cudaMemcpyDeviceToDevice));
        CUDACHECK(cudaMemcpy(d_sendbuff, &d_recvbuff[rank * count], (world_size - rank) * count, cudaMemcpyDeviceToDevice));

        int *rank_r_reps = new int[world_size * w * sizeof(int)];
        for (int i = 0; i < world_size; i++)
        {
            std::vector<int> r_rep = convert10tob(w, i, radix);
            std::memcpy(&rank_r_reps[i * w], r_rep.data(), w * sizeof(int));
        }

        int sent_blocks[nlpow];
        int di = 0;
        int ci = 0;

        char *d_tempbuff;
        CUDACHECK(cudaMalloc((void **)&d_tempbuff, nlpow * count));

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
                        CUDACHECK(cudaMemcpy(&d_tempbuff[count * ci++], &d_sendbuff[count * i], count, cudaMemcpyDeviceToDevice));
                    }
                }

                int distance = z * myPow(radix, x);
                int recv_proc = (rank - distance + world_size) % world_size;
                int send_proc = (rank + distance) % world_size;
                long long comm_size = di * count;

                NCCLCHECK(ncclGroupStart());
                NCCLCHECK(ncclSend(d_tempbuff, comm_size, ncclChar, send_proc, comm, stream));
                NCCLCHECK(ncclRecv(d_recvbuff, comm_size, ncclChar, recv_proc, comm, stream));
                NCCLCHECK(ncclGroupEnd());

                for (int i = 0; i < di; i++)
                {
                    long long offset = sent_blocks[i] * count;
                    CUDACHECK(cudaMemcpy(d_sendbuff + offset, d_recvbuff + (count * i), count, cudaMemcpyDeviceToDevice));
                }
            }
        }

        for (int i = 0; i < world_size; i++)
        {
            int index = (rank - i + world_size) % world_size;
            CUDACHECK(cudaMemcpy(&d_recvbuff[index * count], &d_sendbuff[count * i], count, cudaMemcpyDeviceToDevice));
        }
        /* Bruck End */

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        cudaMemcpy(h_recvbuff, d_recvbuff, bytes, cudaMemcpyDeviceToHost);
        bool passed = true;
        for (int i = 0; i < bytes; i++)
        {
            if (h_recvbuff[i] != h_verifybuff[i])
            {
                passed = false;
            }
        }

        if (passed)
        {
            std::cout << "Rank " << rank << " passed" << std::endl;
        }
        else
        {
            std::cout << "Rank " << rank << " failed:\t[";
            print_buffer(rank, "recvbuff", d_recvbuff, count, bytes);
        }

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        delete[] h_sendbuff;

        CUDACHECK(cudaStreamSynchronize(stream));

        CUDACHECK(cudaFreeAsync(d_sendbuff, stream));
        CUDACHECK(cudaFreeAsync(d_recvbuff, stream));
        CUDACHECK(cudaFreeAsync(d_tempbuff, stream));

        CUDACHECK(cudaStreamSynchronize(stream));
    }

    MPICHECK(MPI_Finalize());
    return 0;
}