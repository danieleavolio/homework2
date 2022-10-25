#include <stdint.h>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define CHANNEL_NUM 3

#include "stb_image.h"
#include "stb_image_write.h"

using namespace std;

__global__ void colorToGrey(unsigned char *Pout, unsigned char *Pin, int width,
                            int height)
{
    // Pout and Pin point to 1 dimensional array
    int Col = (threadIdx.x + blockIdx.x * blockDim.x) * CHANNEL_NUM;
    int Row = (threadIdx.y + blockIdx.y * blockDim.y) * CHANNEL_NUM;
    // get 1D coordinate for the grayscale image
        int greyOffset = (Row * width) + Col;
        int rgb = (Row * width) + Col;
        // one can think of the RGB image having
        // CHANNEL times columns than the grayscale image
        // channel is 3
        // those indexes help to access to the correct cell for the pixel
        unsigned char r = Pin[rgb];     // red value for pixel
        unsigned char g = Pin[rgb+ 1]; // green value for pixel
        unsigned char b = Pin[rgb+ 2]; // blue value for pixel
        // perform the rescaling and store it
        // We multiply by floating point constants
        Pout[greyOffset] = 0.21 * r + 0.71 * g + 0.07 * b;
}
struct Pixel
{
    unsigned char r, g, b, a;
};

void checkCudaError()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

void convertImageToGrayCPU(unsigned char *rgb_image, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            // for every pixel
            Pixel *ptrPixel = (Pixel *)&rgb_image[y * width * 3 + 3 * x];
            unsigned char pixelValue = (unsigned char)ptrPixel->r * 0.2126f + ptrPixel->g * 0.7152f + ptrPixel->b * 0.0722f;
            ptrPixel->r = pixelValue;
            ptrPixel->g = pixelValue;
            ptrPixel->b = pixelValue;
            ptrPixel->a = 255;
        }
    }
}

int main()
{
    int width = 100, height = 100, bpp;

    unsigned char *bw_image;
    unsigned char *rgb_image;
    unsigned char *cpu_rgb_image;

    int size = width * height * sizeof(unsigned char) * CHANNEL_NUM;
    // cudaMallocManaged(&rgb_image, size);
    cout << "Alloco la memoria necessaria \n";
    cudaMalloc(&rgb_image, size);
    cudaMalloc(&cpu_rgb_image, size);
    cudaMallocManaged(&bw_image, size);
    cpu_rgb_image = stbi_load("image.png", &width, &height, &bpp, 3);
    int block_size = 32;
    int number_of_blocks = (width * height * CHANNEL_NUM) / block_size;
    cout << "Copio sulla GPU \n";
    cudaMemcpy(rgb_image, cpu_rgb_image, size, cudaMemcpyHostToDevice);
    cout << "Chiamo la funzione kernel \n";
    colorToGrey<<<number_of_blocks, block_size>>>(bw_image, rgb_image, width, height);
    cudaDeviceSynchronize();
    cout << "Controllo gli errori \n";
    cout << "Copio sulla CPU\n";
    checkCudaError();
    stbi_write_png("image_bw.png", width, height, CHANNEL_NUM, bw_image, width * CHANNEL_NUM);
    cout << "Libero tutto \n";
    cudaFree(bw_image);
    cudaFree(rgb_image);
    cudaFree(cpu_rgb_image);

    return 0;
}