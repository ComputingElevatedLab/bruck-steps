#include <cmath>
#include <iostream>
#include <iomanip>

#include "mpi.h"
#include "nccl.h"

#include "../../common/error-catch.cu"
#include "../../common/error-catch.h"
#include "../../common/hostname.cu"

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
        std::cout << "\nspreadout-nccl" << std::endl;
    }

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Host variables
    char *h_sendbuff;

    // Device variables
    char *d_sendbuff;
    char *d_recvbuff;

    for (int count = 2; count <= 4; count *= 2)
    {
        int bytes = count * world_size;
        h_sendbuff = new char[bytes];
        std::fill_n(h_sendbuff, bytes, rank);

        CUDACHECK(cudaMalloc((void **)&d_sendbuff, bytes));
        CUDACHECK(cudaMalloc((void **)&d_recvbuff, bytes));
        CUDACHECK(cudaMemcpy(d_sendbuff, h_sendbuff, bytes, cudaMemcpyHostToDevice));

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        CUDACHECK(cudaStreamSynchronize(stream));

        print_buffer(rank, "sendbuff", d_sendbuff, count, count);

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        ncclGroupStart();
        for (int i = 0; i < world_size; i++)
        {
            int src = (rank + i) % world_size;
            std::cout << "Count " << count << ": Rank " << rank << " receiving from rank " << src << std::endl;
            ncclRecv(&d_recvbuff[src * count], count, ncclChar, src, comm, stream);
        }

        for (int i = 0; i < world_size; i++)
        {
            int dst = (rank - i + world_size) % world_size;
            std::cout << "Count " << count << ": Rank " << rank << " sending to rank " << dst << std::endl;
            ncclSend(&d_sendbuff[dst * count], count, ncclChar, dst, comm, stream);
        }
        ncclGroupEnd();

        CUDACHECK(cudaStreamSynchronize(stream));
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        print_buffer(rank, "recvbuff", d_recvbuff, count, bytes);

        CUDACHECK(cudaFree(d_sendbuff));
        CUDACHECK(cudaFree(d_recvbuff));
    }

    MPICHECK(MPI_Finalize());
    return 0;
}