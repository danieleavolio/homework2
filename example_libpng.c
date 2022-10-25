#include <libpng16/png.h>

//A support structure for pngs
png_infop info_ptr;

//The image width
png_uint_32 WIDTH;

//The image height
png_uint_32 HEIGHT;

//The pixels matrix
//It has HEIGHT rows and WIDTH*CHANNELS columns
png_bytepp row_pointers; 

//How many cells each pixel occupies in the matrix
int CHANNELS = 3;

//Reads an image's pixel from the file in file_name
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

    //If images have transparency, each pixel will occupy 4 cells in the matrix (rgba)
    //Otherwise it only occupies 3 (rgb)
    if(color_type == PNG_COLOR_TYPE_RGB)
        CHANNELS = 3;
    else if (color_type == PNG_COLOR_TYPE_RGB_ALPHA)
        CHANNELS = 4;
        
    png_destroy_read_struct(&png_ptr, NULL, NULL); 
    fclose(fp);
}

//Writes an image's pixel to the file file_name
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

int main(int argc, char* argv[]) {
    read_png(argv[1]);
    write_png(argv[2]);
}