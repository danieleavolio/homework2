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

__global__ void sharpenConvolution(unsigned char *Pout, unsigned char *Pin, int w,
                                   int h, int channels)
{
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    // printf("Row: %d - Col: %d  \n",Row,Col);
    int sharpen[3][3] = {{0, -1, 0},
                         {-1, 5, -1},
                         {0, -1, 0}};

    int r = 0, g = 0, b = 0, a = 0;

    if (Col < w && Row < h)
    {
        int Pout_offset = Row * w + Col;
        int Pin_offset = Pout_offset * channels;

        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                int neighbor_row = Row + i;
                int neighbor_col = Col + j;
                if (neighbor_row > -1 && neighbor_row < h && neighbor_col > -1 && neighbor_col < w)
                {
                    int neighbor_offset = (neighbor_row * w + neighbor_col) * channels;
                    r += Pin[neighbor_offset] * sharpen[i + 1][j + 1];
                    g += Pin[neighbor_offset + 1] * sharpen[i + 1][j + 1];
                    b += Pin[neighbor_offset + 2] * sharpen[i + 1][j + 1];
                    if (channels == 4)
                        a += Pin[neighbor_offset + 3] * sharpen[i + 1][j + 1];
                }
            }
        }

        // printf("%d %d %d %d \n",r,g,b,a);
        Pout[Pout_offset * channels + 0] = r;
        Pout[Pout_offset * channels + 1] = g;
        Pout[Pout_offset * channels + 2] = b;
        if (channels == 4)
            Pout[Pout_offset * channels + 3] = a;
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
    cudaMallocManaged(&conv_image, size);
    cudaMallocManaged(&rgb_image, size);
    dim3 block_size(32, 32);
    dim3 number_of_blocks(ceil(width / block_size.x), ceil(height / block_size.y));
    cout << "Copio sulla GPU \n";
    cudaMemcpy(rgb_image, cpu_rgb_image, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cout << "Chiamo la funzione kernel \n";
    cout << width << " " << height << " " << channel << endl;
    sharpenConvolution<<<number_of_blocks, block_size>>>(conv_image, rgb_image, width, height, channel);
    cudaDeviceSynchronize();
    cout << "Controllo gli errori \n";
    checkCudaError();
    cout << "Scrivo l'immagine \n";
    stbi_write_jpg(output_name, width, height, channel, conv_image, 100);
    cout << "Libero tutto \n";
    cudaFree(conv_image);
    cudaFree(rgb_image);
    cudaFree(cpu_rgb_image);

    delete conv_image;
    delete cpu_rgb_image;
    delete rgb_image;

    return 0;
}