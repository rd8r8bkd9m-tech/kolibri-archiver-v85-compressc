#ifndef KOLIBRI_HUFFMAN_ANS_H
#define KOLIBRI_HUFFMAN_ANS_H

/*
 * Huffman + ANS сжатие для Kolibri
 *
 * Дополняет существующее Mathematical+LZ77+RLE сжатие
 * для конкуренции с gzip/zstd на общих данных.
 */

#include <stddef.h>
#include <stdint.h>

typedef enum {
    KOLIBRI_ENTROPY_HUFFMAN = 0,  /* Классическое Huffman-кодирование */
    KOLIBRI_ENTROPY_ANS     = 1,  /* tANS — быстрее, сжатие ~arithmetic */
    KOLIBRI_ENTROPY_AUTO    = 2,  /* Автовыбор лучшего метода */
} KolibriCompressMethod;

/* --- Huffman --- */
size_t huffman_compress(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_capacity
);

size_t huffman_decompress(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_capacity
);

/* --- ANS (Asymmetric Numeral Systems) --- */
size_t ans_compress(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_capacity
);

size_t ans_decompress(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_capacity
);

/* --- Unified API --- */
size_t kolibri_entropy_compress(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_capacity,
    KolibriCompressMethod method
);

size_t kolibri_entropy_decompress(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_capacity
);

#endif /* KOLIBRI_HUFFMAN_ANS_H */
