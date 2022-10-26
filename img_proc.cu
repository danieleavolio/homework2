#include <stdint.h>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define CHANNEL_NUM 3

#include "stb_image.h"
#include "stb_image_write.h"

using namespace std;

__global__ void colorToGrey(unsigned char *Pout, unsigned char *Pin, int width,
                            int height, int channels)
{
    // Pout and Pin point to 1 dimensional array
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int size = width * height * channels;
    // for (int i = 0; i < size; i+=channels){
    int media = (Pin[i] + Pin[i + 1] + Pin[1 + 2]) / 3;
    Pout[i] = Pout[i + 1] = Pout[i + 2] = media;
    if (channels == 4) Pout[i+3] = Pin[i+3];
    //}
}

void serialeBw(unsigned char *Pout, unsigned char *Pin, int width, int height, int channels)
{

    //int i = blockDim.x * blockIdx.x + threadIdx.x;
//
    //int size = width * height * channels;
    //// for (int i = 0; i < size; i+=channels){
    //int media = (Pin[i] + Pin[i + 1] + Pin[1 + 2]) / 3;
    //Pout[i] = Pout[i + 1] = Pout[i + 2] = media;
    ////}
    //stbi_write_png("image_bw.png", width, height, channels, Pout, width * channels);
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
    int width, height, channel;

    unsigned char *bw_image;
    unsigned char *rgb_image;
    unsigned char *cpu_rgb_image;
    cpu_rgb_image = stbi_load("godrick.jpg", &width, &height, &channel, 0);
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
    colorToGrey<<<number_of_blocks, block_size>>>(bw_image, rgb_image, width, height, channel);
    cudaDeviceSynchronize();
    cout << "Controllo gli errori \n";
    cout << "Copio sulla CPU\n";
    checkCudaError();
    // serialeBw(bw_image, cpu_rgb_image, width, height, channel);
    stbi_write_png("image_bw.png", width, height, channel, bw_image, width * channel);
    cout << "Libero tutto \n";
    cudaFree(bw_image);
    cudaFree(rgb_image);
    cudaFree(cpu_rgb_image);

    return 0;
}