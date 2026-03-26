#ifndef UTILS_H
#define UTILS_H

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#ifndef TFLITE_DCHECK_GE
#define TFLITE_DCHECK_GE(x, y) assert((x) >= (y))
#endif
#ifndef TFLITE_DCHECK_EQ
#define TFLITE_DCHECK_EQ(x, y) assert((x) == (y))
#endif
#ifndef TFLITE_DCHECK_LE
#define TFLITE_DCHECK_LE(x, y) assert((x) <= (y))
#endif
#endif