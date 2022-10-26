#include <stdint.h>
#include <iostream>
#include <string>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define CHANNEL_NUM 3

#include "stb_image.h"
#include "stb_image_write.h"

using namespace std;

void checkCudaError()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

__global__ void kernelFunction(unsigned char *Pout, unsigned char *Pin, int width,
                            int height, int channels)
{
    // Pout and Pin point to 1 dimensional array
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int size = width * height * channels;
    // for (int i = 0; i < size; i+=channels){
    // int media = (Pin[i] + Pin[i + 1] + Pin[i+2]) / 3;
    // Pout[i] = Pout[i + 1] = Pout[i + 2] = media;
    // if (channels == 4) Pout[i+3] = Pin[i+3];
    //}
}


int main(int argc, char **argv)
{
    int width, height, channel;

    unsigned char *bw_image;
    unsigned char *rgb_image;
    unsigned char *cpu_rgb_image;

    const char *input_name = argv[1];
    const char *output_name = argv[2];

    if (argc != 3)
    {
        cout << "There was an error. Insert the correct number of arguments" << endl;
        exit(EXIT_FAILURE);
    }
    cpu_rgb_image = stbi_load(input_name, &width, &height, &channel, 0);
    int size = width * height * sizeof(unsigned char) * channel;
    cout << "Alloco la memoria necessaria \n";
    cudaMalloc(&rgb_image, size);
    cudaMallocManaged(&bw_image, size);
    int block_size = 32;
    int number_of_blocks = ceil((width * height * channel) / block_size);
    cout << "Copio sulla GPU \n";
    cudaMemcpy(rgb_image, cpu_rgb_image, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cout << "Chiamo la funzione kernel \n";
    cout << width << " " << height << " " << channel << endl;
    kernelFunction<<<number_of_blocks, block_size>>>(bw_image, rgb_image, width, height, channel);
    cudaDeviceSynchronize();
    cout << "Controllo gli errori \n";
    checkCudaError();
    cout << "Scrivo l'immagine \n";
    stbi_write_jpg(output_name, width, height, channel, bw_image, 100);
    cout << "Libero tutto \n";
    cudaFree(bw_image);
    cudaFree(rgb_image);
    cudaFree(cpu_rgb_image);

    delete bw_image;
    delete cpu_rgb_image;
    delete rgb_image;

    return 0;
}