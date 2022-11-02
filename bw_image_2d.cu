#include <stdint.h>
#include <iostream>
#include <string>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define CHANNEL_NUM 3

#include "libs/stb_image.h"
#include "libs/stb_image_write.h"

using namespace std;

void checkCudaError()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
};

/*

    512 *

*/

__global__ void toGray(unsigned char *Pout, unsigned char *Pin, int w,
                               int h, int channels)
{
    // Pout and Pin point to 1 dimensional array
    int col = (blockDim.x * blockIdx.x) + threadIdx.x;
    int row = (blockDim.y * blockIdx.y) + threadIdx.y;
    if (col < w && row < h)
    {
        int correctIndex = (row * w) + col;
        int rgbOffset = correctIndex * channels;
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset+1];
        unsigned char b = Pin[rgbOffset+2];

		Pout[rgbOffset] = 0.21 * r + 0.71 * g + 0.07 * b;
		Pout[rgbOffset+1] = 0.21 * r + 0.71 * g + 0.07 * b;
		Pout[rgbOffset+2] = 0.21 * r + 0.71 * g + 0.07 * b;
    }
}

int main(int argc, char **argv)
{
    int width, height, channel;

    unsigned char *conv_image;
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
    cudaMallocManaged(&conv_image, size);
    dim3 block_size(32, 32);
    dim3 number_of_blocks(width / block_size.x+1, height/block_size.y+1);
    cout << "Copio sulla GPU \n";
    cudaMemcpy(rgb_image, cpu_rgb_image, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cout << "Chiamo la funzione kernel \n";
    cout << width << " " << height << " " << channel << endl;
    toGray<<<number_of_blocks, block_size>>>(conv_image, rgb_image, width, height, channel);
    cudaDeviceSynchronize();
    cout << "Controllo gli errori \n";
    checkCudaError();
    cout << "Scrivo l'immagine \n";
    stbi_write_jpg(output_name, width, height, channel, conv_image, 100 );
    cout << "Libero tutto \n";
    cudaFree(conv_image);
    cudaFree(rgb_image);
    cudaFree(cpu_rgb_image);

    delete conv_image;
    delete cpu_rgb_image;
    delete rgb_image;

    return 0;
}