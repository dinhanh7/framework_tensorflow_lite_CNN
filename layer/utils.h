#ifndef UTILS_H
#define UTILS_H

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
void read_int_array(const char* filename, int32_t* buffer, int size) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        if (fscanf(f, "%d", &buffer[i]) != 1) {
            printf("Error reading int at index %d from %s\n", i, filename);
            break;
        }
    }
    fclose(f);
}

// Helper function to read float array from text file
void read_float_array(const char* filename, float* buffer, int size) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        if (fscanf(f, "%f", &buffer[i]) != 1) {
            printf("Error reading float at index %d from %s\n", i, filename);
            break;
        }
    }
    fclose(f);
}

// Helper function to read int8 array from text file
void read_int8_array(const char* filename, int8_t* buffer, int size) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }
    int temp;
    for (int i = 0; i < size; i++) {
        if (fscanf(f, "%d", &temp) != 1) {
             // Try reading as float and casting if int fails? 
             // Usually file has integer values like "-12"
             // But sometimes numpy text save uses float format "0.000"
             float ftemp;
             rewind(f); // Just in case, but complicated.
             // Let's assume int format first.
             printf("Error reading int8 at index %d from %s\n", i, filename);
             break;
        }
        buffer[i] = (int8_t)temp;
    }
    fclose(f);
}
int count_elements(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) return -1;
    int count = 0;
    int temp;
    while (fscanf(f, "%d", &temp) == 1) count++;
    fclose(f);
    return count;
}
void write_int8_file(const char* filename, int8_t* data, int size) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    for (int i = 0; i < size; ++i) {
        fprintf(f, "%d\n", data[i]);
    }
    fclose(f);
    printf("Wrote output to %s\n", filename);
}

#endif