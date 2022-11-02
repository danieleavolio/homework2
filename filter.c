#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

int min(int a,int b){
    return a < b ? a : b;
}

int max(int a, int b){
    return a > b ? a : b;
}

void conv_filter(uint8_t* in, uint8_t* out, int w, int h, int channels, int* filter, int fdim){
    int left_border,right_border, up_border, bot_border;
    int sum_r, sum_g,sum_b;
    int ray = (fdim - 1)/2;
    int top, left, right, bot;
    //scorro le righe ...
    for(int r = 0; r < h; r++){
        //define offsets to conv kernel ...
        top = max(ray * -1, r * -1);
        bot = min(ray, h - r);
        for(int c = 0; c < w; c++){
            left = max(ray * -1, c * -1);
            right = min(ray, w -c);

            sum_r = sum_b = sum_g = 0; 

            //scorro la matrice del filtro sul pixel ... 
            int done = 0;
            for(int kr = top; kr <= bot; kr++){
                for(int kc = left; kc <= right; kc++){

                    sum_r += in[(r + kr) * channels * w + (c + kc) * channels] * filter[done];
                    sum_g += in[(r + kr) * channels * w + (c + kc) * channels + 1] * filter[done];
                    sum_b += in[(r + kr) * channels * w + (c + kc) * channels + 2] * filter[done];
                    done++;
                }
            }

            //printf("%d %d %d\n", sum_r,sum_g,sum_b);

            out[r * channels * w + c * channels] = min(max(0,sum_r) ,255);
            out[r * channels * w + c * channels + 1] = min(max(0,sum_g),255);
            out[r * channels * w + c * channels + 2] = min(max(0,sum_b), 255);

        }
    }

}


int main(int argc, char **argv) {
    int width, height, bpp;

    printf(argv[1]);
    uint8_t* rgb_image = stbi_load(argv[1], &width, &height, &bpp, 3);
    
    uint8_t * rgb_out = (uint8_t*) malloc(width*height*3 * sizeof(uint8_t));

    int fdim = 3;
    uint8_t* filter = (uint8_t*) malloc(fdim * fdim * sizeof(uint8_t));
    filter[0] = 0; filter[1] = -1; filter[2] = 0;   
    filter[3] = -1; filter[4] = 5;  filter[5] = -1;
    filter[6] = 0; filter[7] = -1; filter[8] = 0;

    uint8_t * dev_in, dev_out, dev_filter;
    //cuda installations ...
    cudaMalloc((void**)dev_in, width*height* 3 * sizeof(uint8_t));
    cudaMalloc((void**)dev_out, width*height* 3 * sizeof(uint8_t));
    cudaMalloc((void**)dev_filter, fdim * fdim * sizeof(uint8_t));

    cudaMemcpy(dev_in, rgb_image,width*height* 3 * sizeof(uint8_t),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_filter, filter,fdim*fdim * sizeof(uint8_t) ,cudaMemcpyHostToDevice);

    int block_size = 32;
    int block_num = ceil( (width * height)/ block_size);

    conv_filter<<<block_size, block_num>>>(dev_in,dev_out, width,height,3,dev_filter,fdim);

    cudaMemcpy(dev_out, rgb_out,width*height* 3 * sizeof(uint8_t),cudaMemcpyDeviceToHost);


    char outp[80];
    snprintf(outp,sizeof(outp),"%s%s", "filt_", argv[1]); 
    stbi_write_png(outp, width, height, 3, rgb_out, width*3);

    cudaFree(dev_in);
    cudaFree(dev_filter);
    cudaFree(dev_out)
    stbi_image_free(rgb_image);
    return 0;
}
//http://www.libpng.org/pub/png/libpng.html