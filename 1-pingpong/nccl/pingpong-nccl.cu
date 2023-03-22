#include <iostream>
#include <iomanip>

#include "mpi.h"
#include "nccl.h"

#include "../../common/error-catch.cu"
#include "../../common/error-catch.h"
#include "../../common/hostname.cu"

void print_buffer(int rank, std::string name, int *d_buffer, int count)
{
    int *h_buffer = new int[count];
    cudaMemcpy(h_buffer, d_buffer, count * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << std::fixed << "Count " << count << ": Rank " << rank << " " << name << " buffer: [";
    for (int i = 0; i < count; i++)
    {
        std::cout << " " << h_buffer[i] << " ";
    }
    std::cout << "]" << std::endl;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
        std::cout << "\npingpong-nccl" << std::endl;
    }

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Host variables
    int *h_sendbuff;
    int *h_recvbuff;

    // Device variables
    int *d_sendbuff;
    int *d_recvbuff;

    for (int count = 2; count <= 4; count *= 2)
    {
        int bytes = count * sizeof(int);

        h_sendbuff = new int[count];
        h_recvbuff = new int[count];
        std::fill_n(h_sendbuff, count, rank);
        std::fill_n(h_recvbuff, count, -1);

        CUDACHECK(cudaMalloc((void **)&d_sendbuff, bytes));
        CUDACHECK(cudaMalloc((void **)&d_recvbuff, bytes));
        CUDACHECK(cudaMemcpy(d_sendbuff, h_sendbuff, bytes, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(d_recvbuff, h_recvbuff, bytes, cudaMemcpyHostToDevice));

        int src = 0;
        int dst = world_size - 1;

        MPI_Barrier(MPI_COMM_WORLD);
        CUDACHECK(cudaStreamSynchronize(stream));

        if (rank == src)
        {
            print_buffer(rank, "sendbuff", d_sendbuff, count);
        }

        ncclGroupStart();
        if (rank == dst)
        {
            std::cout << "Count " << count << ": Rank " << rank << " receiving from rank 0" << std::endl;
            NCCLCHECK(ncclRecv(d_recvbuff, count, ncclInt, 0, comm, stream));
        }

        if (rank == src)
        {
            std::cout << "Count " << count << ": Rank " << rank << " sending to rank " << dst << std::endl;
            NCCLCHECK(ncclSend(d_sendbuff, count, ncclInt, dst, comm, stream));
        }
        ncclGroupEnd();

        CUDACHECK(cudaStreamSynchronize(stream));
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        if (rank == dst)
        {
            print_buffer(rank, "recvbuff", d_recvbuff, count);
        }

        CUDACHECK(cudaFree(d_sendbuff));
        CUDACHECK(cudaFree(d_recvbuff));
    }

    MPI_Finalize();
    return 0;
}