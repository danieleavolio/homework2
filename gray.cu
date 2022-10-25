#include <libpng16/png.h>

png_bytepp row_pointers; 
png_infop info_ptr;
png_uint_32 WIDTH;
png_uint_32 HEIGHT;
int CHANNELS = 3;

void read_png(char *file_name)
{
    FILE *fp = fopen(file_name, "rb");
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    row_pointers = png_get_rows(png_ptr, info_ptr);

    int color_type;
    png_get_IHDR(png_ptr, info_ptr, &WIDTH, &HEIGHT, NULL, &color_type, NULL, NULL, NULL);

    if(color_type == PNG_COLOR_TYPE_RGB)
        CHANNELS = 3;
    else if (color_type == PNG_COLOR_TYPE_RGB_ALPHA)
        CHANNELS = 4;
        
    png_destroy_read_struct(&png_ptr, NULL, NULL); 
    fclose(fp);
}

void write_png(char *file_name)
{
    FILE *fp = fopen(file_name, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_init_io(png_ptr, fp);
    png_set_rows(png_ptr, info_ptr, row_pointers);
    png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__global__
void turn_gray_kernel(unsigned char* d_pixels, unsigned char* d_pixels_gray, int HEIGHT, int WIDTH, int CHANNELS) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //Check if thread is within the image boundaries
    if(row < HEIGHT && col < WIDTH) {

        //Get rgb value for the current pixel
        unsigned char r = d_pixels[row*WIDTH*CHANNELS + col*CHANNELS];
        unsigned char g = d_pixels[row*WIDTH*CHANNELS + col*CHANNELS + 1];
        unsigned char b = d_pixels[row*WIDTH*CHANNELS + col*CHANNELS + 2];

        //Compute grayscale value with luminosity formula
        unsigned char gray = r * 0.21 + g * 0.72 + b * 0.07;

        //Assign gray value to all 3 rgb values of the output pixels array
        d_pixels_gray[row*WIDTH + col] = gray;
    }
}

void turn_gray() {

    unsigned char* linearized_pixels, *linearized_pixels_gray;

    linearized_pixels = (unsigned char*) malloc(HEIGHT*WIDTH*CHANNELS * sizeof(unsigned char));
    linearized_pixels_gray = (unsigned char*) malloc(HEIGHT*WIDTH * sizeof(unsigned char));

    //Linearizing the image pixels into an array for cuda
    for(int i = 0; i < HEIGHT; ++i) {
        for(int j = 0; j < WIDTH*CHANNELS; j += CHANNELS) {
            linearized_pixels[i*WIDTH*CHANNELS + j] = row_pointers[i][j];
            linearized_pixels[i*WIDTH*CHANNELS + j + 1] = row_pointers[i][j + 1];
            linearized_pixels[i*WIDTH*CHANNELS + j + 2] = row_pointers[i][j + 2];
        }
    }

    unsigned char* d_pixels, *d_pixels_gray;

    cudaMalloc((void**) &d_pixels, HEIGHT*WIDTH*CHANNELS * sizeof(unsigned char));
    cudaMalloc((void**) &d_pixels_gray, HEIGHT*WIDTH * sizeof(unsigned char));

    //Copying the pixels to the gpu
    cudaMemcpy(d_pixels, linearized_pixels, HEIGHT*WIDTH*CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice);

    //Defining block size and number of blocks
    dim3 block_size(32, 32, 1);
    dim3 number_of_blocks(WIDTH / block_size.x + 1, HEIGHT / block_size.y + 1, 1);

    //Invoking the kernel that turns pixels into a grayscale
    turn_gray_kernel<<<number_of_blocks, block_size>>>(d_pixels, d_pixels_gray, HEIGHT, WIDTH, CHANNELS);

    //Copying the grayscale pixels from the gpu to the host
    cudaMemcpy(linearized_pixels_gray, d_pixels_gray, HEIGHT*WIDTH * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    //Copying the gray pixels into the row of the image
    for(int i = 0; i < HEIGHT; ++i) {
        for(int j = 0; j < WIDTH; ++j) {
            row_pointers[i][j*CHANNELS] = linearized_pixels_gray[i*WIDTH + j];
            row_pointers[i][j*CHANNELS + 1] = linearized_pixels_gray[i*WIDTH + j];
            row_pointers[i][j*CHANNELS + 2] = linearized_pixels_gray[i*WIDTH + j];
        }
    }

    free(linearized_pixels);
    free(linearized_pixels_gray);

    cudaFree(d_pixels);
    cudaFree(d_pixels_gray);
}

int main(int argc, char* argv[]) {
    read_png(argv[1]);
    turn_gray();
    write_png(argv[2]);
}