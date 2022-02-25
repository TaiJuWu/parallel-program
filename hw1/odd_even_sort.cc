#include <algorithm>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <cstdio>
#include <mpi.h>
#include <stdlib.h>

using namespace boost::sort::spreadsort;

void printArray(int rank, float *a, int size) {
  printf("rank%d ", rank);
  for (int i = 0; i < size; ++i)
    printf("%f ", a[i]);
  printf("\n");
}

bool checkArrayOrder(int rank, float *a, int size) {
  bool correct = true;
  float errorNum;
  for (int i = 0; i < size - 1; ++i) {
    if (a[i] > a[i + 1]) {
      correct = false;
      errorNum = a[i];
      break;
    }
  }
  if (!correct) {
    printf("rank%d error, errorNum:,%f\n", rank, errorNum);
  }
  return correct;
}

void sender(int rank, float *data, unsigned int partitionSize, int round) {
  float rightMin;
  MPI_Sendrecv(data + partitionSize - 1, 1, MPI_FLOAT, rank + 1, round,
               &rightMin, 1, MPI_FLOAT, rank + 1, round, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  if (rightMin < *(data + partitionSize - 1)) {
    MPI_Send(data, partitionSize, MPI_FLOAT, rank + 1, round, MPI_COMM_WORLD);
    MPI_Recv(data, partitionSize, MPI_FLOAT, rank + 1, round, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
}

void receiver(float *mergeData, unsigned int leftDataSize, int rank,
              float **data, unsigned int partitionSize, float *mergeResult,
              bool *sorted, int round, bool test) {
  float leftMax;
  MPI_Sendrecv(*data, 1, MPI_FLOAT, rank - 1, round, &leftMax, 1, MPI_FLOAT,
               rank - 1, round, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (leftMax > (**data)) {
    MPI_Recv(mergeData, leftDataSize, MPI::FLOAT, rank - 1, round,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::merge(mergeData, mergeData + leftDataSize, *data,
               (*data) + partitionSize, mergeResult);
    // checkArrayOrder(rank, mergeResult, midPartitionSize+partitionSize); //
    // for debug

    if (!test) {
      *sorted = false;
    } else {
      for (int i = 0; i < leftDataSize; ++i) {
        if (mergeResult[i] != mergeData[i]) {
          *sorted = false;
          break;
        }
      }
    }
    // send data to left process
    MPI_Send(mergeResult, leftDataSize, MPI::FLOAT, rank - 1, round,
             MPI_COMM_WORLD);
    *data = mergeResult + leftDataSize;
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // compute partition size for each process
  unsigned long long arraySize = atoll(argv[1]);
  unsigned long long partitionSize = arraySize / size;
  unsigned long long remainder = arraySize - (size * partitionSize);

  if (rank < remainder) {
    partitionSize += 1;
  }

  float *data = new float[partitionSize];
  float *pointerForDelete = data;

  // case is too small, use sequence code
  if (arraySize < 10000) {
    MPI_File inputFile;
    MPI_File outputFile;
    MPI_Group orig_group, new_group;
    MPI_Comm new_comm;
    int process = 0;
    MPI_Comm_group(MPI_COMM_WORLD, &orig_group);
    MPI_Group_incl(orig_group, 1, &process, &new_group);
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

    if (rank == 0) {
      float *array = new float[arraySize];
      MPI_File_open(new_comm, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL,
                    &inputFile);
      MPI_File_read_at(inputFile, 0, array, arraySize, MPI_FLOAT,
                       MPI_STATUS_IGNORE);
      spreadsort(array, array + arraySize);
      // std::sort(array, array+arraySize);
      MPI_File_open(new_comm, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY,
                    MPI_INFO_NULL, &outputFile);
      MPI_File_write_at(outputFile, 0, array, arraySize, MPI_FLOAT,
                        MPI_STATUS_IGNORE);
      MPI_File_close(&inputFile);
      MPI_File_close(&outputFile);
    }
    MPI_Finalize();
    exit(0);
  }
  // read file
  MPI_File inputFile;
  MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL,
                &inputFile);

  if (remainder == 0) {
    MPI_File_read_at(inputFile, sizeof(float) * rank * partitionSize, data,
                     partitionSize, MPI_FLOAT, MPI_STATUS_IGNORE);
  } else {
    if (rank < remainder) {
      MPI_File_read_at(inputFile, sizeof(float) * rank * partitionSize, data,
                       partitionSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    } else {
      MPI_File_read_at(inputFile,
                       sizeof(float) * ((remainder * (partitionSize + 1)) +
                                        (rank - remainder) * partitionSize),
                       data, partitionSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
  }

  MPI_File_close(&inputFile);
  // Odd-even sort
  // use sort for data in each process
  //	qsort(data, partitionSize, sizeof(float), cmpfunc);
  spreadsort(data, data + partitionSize);
  // std::sort(data , data+partitionSize);

  // the data size of left process
  unsigned int leftDataSize = partitionSize;
  if (rank == remainder) {
    leftDataSize = partitionSize + 1;
  }

  // merge 2 process
  unsigned int round = 0;
  float *mergeData = new float[leftDataSize];
  bool sorted = false; // sort is finish or not
  bool global_sorted = false;
  float *mergeResult = new float[leftDataSize + partitionSize];

  while (!global_sorted) {
    sorted = true;
    // # of process is even
    if (size % 2 == 0) {
      // even phase
      if (round % 2 == 0) {
        // process rank is odd, receive data from left
        if (rank % 2 == 1) {
          receiver(mergeData, leftDataSize, rank, &data, partitionSize,
                   mergeResult, &sorted, round, false);
        }
        // process rank is even, send data to right
        else {
          sender(rank, data, partitionSize, round);
        }
      }
      // odd phase
      else {
        // rank is odd, send data to right
        if (rank % 2 == 1 && rank != size - 1) {
          sender(rank, data, partitionSize, round);
        }
        // rank is even ,receive data from left
        else if (rank % 2 == 0 && rank != 0) {
          receiver(mergeData, leftDataSize, rank, &data, partitionSize,
                   mergeResult, &sorted, round, true);
        }
      }
    }
    // # of process is odd
    else {
      // even phase
      if (round % 2 == 0) {
        // process rank is odd, receive data from left
        if (rank % 2 == 1) {
          receiver(mergeData, leftDataSize, rank, &data, partitionSize,
                   mergeResult, &sorted, round, false);
        }
        // process rank is even, send data to right
        else if (rank % 2 == 0 && rank != size - 1) {
          sender(rank, data, partitionSize, round);
        }
      }
      // odd phase
      else {
        // rank is odd, send data to right
        if (rank % 2 == 1) {
          sender(rank, data, partitionSize, round);
        }
        // rank is even ,receive data from left
        else if (rank % 2 == 0 && rank != 0) {
          receiver(mergeData, leftDataSize, rank, &data, partitionSize,
                   mergeResult, &sorted, round, true);
        }
      }
    }
    if (round % 2 == 0 && round != 0) {
      MPI_Allreduce(&sorted, &global_sorted, 1, MPI::BOOL, MPI_LAND,
                    MPI_COMM_WORLD);
    }
    ++round;
  }

  // write file
  MPI_File outputFile;
  int rc =
      MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY,
                    MPI_INFO_NULL, &outputFile);
  if (remainder == 0) {
    MPI_File_write_at(outputFile, sizeof(float) * rank * partitionSize, data,
                      partitionSize, MPI_FLOAT, MPI_STATUS_IGNORE);
  } else {
    if (rank < remainder) {
      MPI_File_write_at(outputFile, sizeof(float) * rank * partitionSize, data,
                        partitionSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    } else {
      MPI_File_write_at(outputFile,
                        sizeof(float) * ((remainder * (partitionSize + 1)) +
                                         (rank - remainder) * partitionSize),
                        data, partitionSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
  }
  MPI_File_close(&outputFile);

  // release all memory
  delete[] pointerForDelete;
  delete[] mergeData;
  delete[] mergeResult;

  MPI_Finalize();
}
