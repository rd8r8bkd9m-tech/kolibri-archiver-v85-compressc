/*
 * Kolibri OS Archiver - Compression Implementation
 * Multi-layer compression with mathematical analysis
 */

#define _GNU_SOURCE  /* CLOCK_MONOTONIC */
#include "kolibri/compress.h"
#include "kolibri/huffman_ans.h"
#include "kolibri/random.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <divsufsort.h>  /* v70: BWT preprocessing via libdivsufsort */

/* v62: SIMD и многопоточность */
#if defined(__SSE2__)
#include <emmintrin.h>
#define KF_USE_SIMD 1
#else
#define KF_USE_SIMD 0
#endif
#if !defined(_WIN32)
#include <pthread.h>
#include <unistd.h>
#define KF_USE_THREADS 1
#else
#define KF_USE_THREADS 0
#endif

/* Disable noisy BWT tracing in release builds (it hurts throughput). */
#ifndef KOLIBRI_TRACE_BWT
#define KOLIBRI_TRACE_BWT 0
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

static size_t kolibri_thread_count(size_t tasks) {
#if KF_USE_THREADS
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    size_t cores = (n > 0) ? (size_t)n : 4u;
    if (cores < 2) cores = 2;
    if (cores > 16) cores = 16; /* Avoid oversubscription on many-core hosts. */
    if (tasks == 0) return 1;
    return MIN(cores, tasks);
#else
    (void)tasks;
    return 1;
#endif
}

/* Magic number for compressed data format */
#define KOLIBRI_COMPRESS_MAGIC 0x4B4C4252 /* "KLBR" */
#define KOLIBRI_LZCM_MAGIC    0x4D4B      /* "KM" — минимальный LZCM заголовок */
#define KOLIBRI_LZCM_HDR_SIZE 5           /* v67: magic(2) + original_size(3) */
#define KOLIBRI_COMPRESS_VERSION 85  /* v85: Turbo RC + BWT/MTF/ZRLE path improvements */

/* Compression header (v66: compact — 16 bytes) */
typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t methods;
    uint32_t original_size;
    uint32_t checksum;
} KolibriCompressHeader;

struct KolibriCompressor {
    uint32_t methods;
    uint8_t *temp_buffer;
    size_t temp_buffer_size;
};

/* Internal helper functions */
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

/* Checksum implementation (CRC32) */
static const uint32_t crc32_table[256] = {
    0x00000000, 0x77073096, 0xee0e612c, 0x990951ba,
    0x076dc419, 0x706af48f, 0xe963a535, 0x9e6495a3,
    0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
    0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91,
    0x1db71064, 0x6ab020f2, 0xf3b97148, 0x84be41de,
    0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
    0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec,
    0x14015c4f, 0x63066cd9, 0xfa0f3d63, 0x8d080df5,
    0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
    0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b,
    0x35b5a8fa, 0x42b2986c, 0xdbbbc9d6, 0xacbcf940,
    0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
    0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116,
    0x21b4f4b5, 0x56b3c423, 0xcfba9599, 0xb8bda50f,
    0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
    0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d,
    0x76dc4190, 0x01db7106, 0x98d220bc, 0xefd5102a,
    0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
    0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818,
    0x7f6a0dbb, 0x086d3d2d, 0x91646c97, 0xe6635c01,
    0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
    0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457,
    0x65b0d9c6, 0x12b7e950, 0x8bbeb8ea, 0xfcb9887c,
    0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
    0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2,
    0x4adfa541, 0x3dd895d7, 0xa4d1c46d, 0xd3d6f4fb,
    0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
    0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9,
    0x5005713c, 0x270241aa, 0xbe0b1010, 0xc90c2086,
    0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
    0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4,
    0x59b33d17, 0x2eb40d81, 0xb7bd5c3b, 0xc0ba6cad,
    0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
    0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683,
    0xe3630b12, 0x94643b84, 0x0d6d6a3e, 0x7a6a5aa8,
    0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
    0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe,
    0xf762575d, 0x806567cb, 0x196c3671, 0x6e6b06e7,
    0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
    0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5,
    0xd6d6a3e8, 0xa1d1937e, 0x38d8c2c4, 0x4fdff252,
    0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
    0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60,
    0xdf60efc3, 0xa867df55, 0x316e8eef, 0x4669be79,
    0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
    0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f,
    0xc5ba3bbe, 0xb2bd0b28, 0x2bb45a92, 0x5cb36a04,
    0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
    0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a,
    0x9c0906a9, 0xeb0e363f, 0x72076785, 0x05005713,
    0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
    0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21,
    0x86d3d2d4, 0xf1d4e242, 0x68ddb3f8, 0x1fda836e,
    0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
    0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c,
    0x8f659eff, 0xf862ae69, 0x616bffd3, 0x166ccf45,
    0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
    0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db,
    0xaed16a4a, 0xd9d65adc, 0x40df0b66, 0x37d83bf0,
    0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
    0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6,
    0xbad03605, 0xcdd70693, 0x54de5729, 0x23d967bf,
    0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
    0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
};

uint32_t kolibri_checksum(const uint8_t *data, size_t size) {
    uint32_t crc = 0xFFFFFFFF;
    if (!data || size == 0) return crc ^ 0xFFFFFFFF;
    /* Unrolled CRC32 — 8 байт за итерацию */
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        crc = (crc >> 8) ^ crc32_table[(crc ^ data[i])   & 0xFF];
        crc = (crc >> 8) ^ crc32_table[(crc ^ data[i+1]) & 0xFF];
        crc = (crc >> 8) ^ crc32_table[(crc ^ data[i+2]) & 0xFF];
        crc = (crc >> 8) ^ crc32_table[(crc ^ data[i+3]) & 0xFF];
        crc = (crc >> 8) ^ crc32_table[(crc ^ data[i+4]) & 0xFF];
        crc = (crc >> 8) ^ crc32_table[(crc ^ data[i+5]) & 0xFF];
        crc = (crc >> 8) ^ crc32_table[(crc ^ data[i+6]) & 0xFF];
        crc = (crc >> 8) ^ crc32_table[(crc ^ data[i+7]) & 0xFF];
    }
    for (; i < size; i++) {
        crc = (crc >> 8) ^ crc32_table[(crc ^ data[i]) & 0xFF];
    }
    return crc ^ 0xFFFFFFFF;
}

/* File type detection */
KolibriFileType kolibri_detect_file_type(const uint8_t *data, size_t size) {
    if (!data || size == 0) {
        return KOLIBRI_FILE_UNKNOWN;
    }

    /* Check for common image formats */
    if (size >= 2 && data[0] == 0xFF && data[1] == 0xD8) {
        return KOLIBRI_FILE_IMAGE; /* JPEG */
    }
    if (size >= 4 && data[0] == 0x89 && data[1] == 'P' && data[2] == 'N' && data[3] == 'G') {
        return KOLIBRI_FILE_IMAGE; /* PNG */
    }
    if (size >= 3 && data[0] == 'G' && data[1] == 'I' && data[2] == 'F') {
        return KOLIBRI_FILE_IMAGE; /* GIF */
    }
    /* BMP */
    if (size >= 2 && data[0] == 'B' && data[1] == 'M') {
        return KOLIBRI_FILE_IMAGE;
    }
    /* WebP */
    if (size >= 12 && data[0] == 'R' && data[1] == 'I' &&
        data[2] == 'F' && data[3] == 'F' &&
        data[8] == 'W' && data[9] == 'E' &&
        data[10] == 'B' && data[11] == 'P') {
        return KOLIBRI_FILE_IMAGE;
    }

    /* Check if text (ASCII + UTF-8 compatible)
     * Подсчитываем символы: ASCII printable, whitespace, а также
     * валидные UTF-8 мультибайтовые последовательности (кириллица, CJK и т.д.) */
    size_t valid_chars = 0;
    size_t check_size = MIN(size, 1024);
    for (size_t i = 0; i < check_size; ) {
        uint8_t c = data[i];
        if ((c >= 32 && c <= 126) || c == '\n' || c == '\r' || c == '\t') {
            /* ASCII printable + whitespace */
            valid_chars++;
            i++;
        } else if (c >= 0xC0 && c <= 0xFD) {
            /* UTF-8 leading byte — определяем длину последовательности */
            int seq_len = 0;
            if ((c & 0xE0) == 0xC0)      seq_len = 2;  /* 110xxxxx — 2 байта */
            else if ((c & 0xF0) == 0xE0)  seq_len = 3;  /* 1110xxxx — 3 байта */
            else if ((c & 0xF8) == 0xF0)  seq_len = 4;  /* 11110xxx — 4 байта */
            else                          seq_len = 2;  /* Fallback */
            /* Проверяем continuation bytes */
            int valid_seq = 1;
            for (int j = 1; j < seq_len && i + j < check_size; j++) {
                if ((data[i + j] & 0xC0) != 0x80) {
                    valid_seq = 0;
                    break;
                }
            }
            if (valid_seq && i + seq_len <= check_size) {
                valid_chars += seq_len; /* Вся последовательность валидна */
                i += seq_len;
            } else {
                i++;  /* Невалидный UTF-8 */
            }
        } else {
            i++; /* Бинарный байт */
        }
    }

    if (check_size > 0 && valid_chars > check_size * 85 / 100) {
        return KOLIBRI_FILE_TEXT;
    }

    return KOLIBRI_FILE_BINARY;
}

/* RLE decompression */
static size_t decompress_rle(const uint8_t *input, size_t input_size,
                             uint8_t *output, size_t output_size) {
    size_t in_pos = 0;
    size_t out_pos = 0;

    while (in_pos < input_size) {
        if (input[in_pos] == 0xFF && in_pos + 1 < input_size) {
            if (input[in_pos + 1] == 0x00) {
                /* Escaped literal 0xFF */
                if (out_pos >= output_size) return 0;
                output[out_pos++] = 0xFF;
                in_pos += 2;
            } else if (in_pos + 2 < input_size) {
                /* RLE sequence */
                uint8_t count = input[in_pos + 1];
                uint8_t value = input[in_pos + 2];
                
                for (uint8_t i = 0; i < count; i++) {
                    if (out_pos >= output_size) return 0;
                    output[out_pos++] = value;
                }
                in_pos += 3;
            } else {
                /* Incomplete sequence at end */
                return 0;
            }
        } else {
            /* Literal byte */
            if (out_pos >= output_size) return 0;
            output[out_pos++] = input[in_pos++];
        }
    }

    return out_pos;
}


/* LZ77 decompression */
static size_t decompress_lz77(const uint8_t *input, size_t input_size,
                              uint8_t *output, size_t output_size) {
    size_t in_pos = 0;
    size_t out_pos = 0;

    while (in_pos < input_size) {
        if (input[in_pos] == 0xFE && in_pos + 1 < input_size) {
            if (input[in_pos + 1] == 0xFF) {
                /* Escaped literal 0xFE (marker followed by 0xFF) */
                if (out_pos >= output_size) return 0;
                output[out_pos++] = 0xFE;
                in_pos += 2;
            } else if (in_pos + 3 < input_size) {
                /* LZ77 sequence - distance is 2 bytes */
                size_t distance = ((size_t)input[in_pos + 1] << 8) | (size_t)input[in_pos + 2];
                size_t length = input[in_pos + 3];
                
                if (out_pos < distance || distance == 0) return 0;
                
                size_t copy_pos = out_pos - distance;
                for (size_t i = 0; i < length; i++) {
                    if (out_pos >= output_size) return 0;
                    output[out_pos++] = output[copy_pos++];
                }
                in_pos += 4;
            } else {
                /* Incomplete sequence */
                return 0;
            }
        } else {
            /* Literal byte */
            if (out_pos >= output_size) return 0;
            output[out_pos++] = input[in_pos++];
        }
    }

    return out_pos;
}

/* Mathematical pattern analysis for additional compression */
/* Supported Formulas:
 * 0: RAW (No change)
 * 1: DELTA_1 (x[i] - x[i-1]) - Linear simple
 * 2: DELTA_2 (x[i] - 2*x[i-1] + x[i-2]) - Linear progressive
 * 3: STRIDE_4 (x[i] - x[i-4]) - 32-bit integers / structs
 * 4: GENERATOR (Procedural Generation from Seed) - "Genome Formula"
 * 5: GENERATOR_CONTINUE (Continue from previous block)
 * 6: GENERATOR_APPROX (Seed + Delta)
 * 8: BWT+MTF (Burrows-Wheeler Transform + Move-to-Front)
 * 9: CONTEXT_PRED (Order-2 context prediction - Kolibri AI)
 * 10: BWT+MTF+CONTEXT (Combined: BWT + MTF + Context prediction)
 */

/* ===================== BWT (Burrows-Wheeler Transform) ===================== */
/* BWT Inverse Transform: Восстановление исходных данных из BWT */
static int bwt_inverse(const uint8_t *L, size_t len, uint8_t *output, uint32_t primary_index) {
    if (!L || !output || primary_index >= (uint32_t)len) return -1;
    
    /* Подсчёт частот */
    uint32_t count[256] = {0};
    for (size_t i = 0; i < len; i++) count[L[i]]++;
    
    /* Кумулятивные частоты */
    uint32_t cumul[256];
    uint32_t sum = 0;
    for (int c = 0; c < 256; c++) {
        cumul[c] = sum;
        sum += count[c];
    }
    
    /* Строим LF-mapping */
    uint32_t *LF = (uint32_t *)malloc(len * sizeof(uint32_t));
    if (!LF) return -1;
    
    uint32_t running[256];
    memcpy(running, cumul, sizeof(cumul));
    for (size_t i = 0; i < len; i++) {
        LF[i] = running[L[i]]++;
    }
    
    /* Восстанавливаем строку обратным обходом цепочки */
    uint32_t idx = primary_index;
    for (size_t i = len; i > 0; i--) {
        output[i - 1] = L[idx];
        idx = LF[idx];
    }
    
    free(LF);
    return 0;
}

/* ===================== MTF (Move-to-Front Transform) ===================== */

/* MTF Decode: Обратное преобразование рангов в байты */
static void mtf_decode(uint8_t *data, size_t len) {
    uint8_t table[256];
    for (int i = 0; i < 256; i++) table[i] = (uint8_t)i;
    
    for (size_t i = 0; i < len; i++) {
        uint8_t rank = data[i];
        uint8_t ch = table[rank];
        
        data[i] = ch;
        memmove(table + 1, table, rank);
        table[0] = ch;
    }
}

/* ===================== Context Prediction v50 (Kolibri AI) ===================== */
/* 
 * v50: \u0421\u043c\u0435\u0448\u0430\u043d\u043d\u043e\u0435 \u043f\u0440\u0435\u0434\u0441\u043a\u0430\u0437\u0430\u043d\u0438\u0435 Order-1 + Order-2 + \u0445\u0435\u0448\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u044b\u0439 Order-3.
 * \u041a\u0430\u0436\u0434\u044b\u0439 \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442 \u0440\u0430\u0431\u043e\u0442\u0430\u0435\u0442 \u043d\u0435\u0437\u0430\u0432\u0438\u0441\u0438\u043c\u043e \u2014 \u043f\u0440\u0435\u0434\u0441\u043a\u0430\u0437\u0430\u043d\u0438\u044f \u0441\u043c\u0435\u0448\u0438\u0432\u0430\u044e\u0442\u0441\u044f \u0441 \u0432\u0435\u0441\u0430\u043c\u0438.
 * Order-3 \u043f\u0440\u0435\u0434\u0441\u043a\u0430\u0437\u044b\u0432\u0430\u0435\u0442 \u043b\u0443\u0447\u0448\u0435 \u043d\u0430 \u0441\u0442\u0440\u0443\u043a\u0442\u0443\u0440\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u044b\u0445 \u0434\u0430\u043d\u043d\u044b\u0445,
 * \u043d\u043e Order-1/2 \u043e\u0441\u0442\u0430\u044e\u0442\u0441\u044f \u043d\u0430\u0434\u0451\u0436\u043d\u044b\u043c fallback.
 */

#define CTX_TABLE_SIZE 65536  /* 256 * 256 = Order-2 context */
#define CTX3_TABLE_SIZE 262144 /* 2^18 = \u0445\u0435\u0448\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u044b\u0439 Order-3 context */
#define CTX3_MASK (CTX3_TABLE_SIZE - 1)

static void context_predict_decode(const uint8_t *input, size_t len, uint8_t *output) {
    uint8_t *pred1 = (uint8_t *)calloc(256, 1);
    uint8_t *pred2 = (uint8_t *)calloc(CTX_TABLE_SIZE, 1);
    uint8_t *pred3 = (uint8_t *)calloc(CTX3_TABLE_SIZE, 1);
    uint8_t *hit1  = (uint8_t *)calloc(256, 1);
    uint8_t *hit2  = (uint8_t *)calloc(CTX_TABLE_SIZE, 1);
    uint8_t *hit3  = (uint8_t *)calloc(CTX3_TABLE_SIZE, 1);
    
    if (!pred1 || !pred2 || !pred3 || !hit1 || !hit2 || !hit3) {
        memcpy(output, input, len);
        free(pred1); free(pred2); free(pred3);
        free(hit1); free(hit2); free(hit3);
        return;
    }
    
    if (len > 0) output[0] = input[0];
    if (len > 1) output[1] = input[1];
    if (len > 2) output[2] = input[2];
    
    for (size_t i = 3; i < len; i++) {
        uint8_t ctx1_key = output[i-1];
        uint16_t ctx2_key = ((uint16_t)output[i-2] << 8) | (uint16_t)output[i-1];
        uint32_t ctx3_key = (((uint32_t)output[i-3] * 65599U) ^
                             ((uint32_t)output[i-2] * 257U) ^
                             (uint32_t)output[i-1]) & CTX3_MASK;
        
        int p1 = (int)pred1[ctx1_key];
        int p2 = (int)pred2[ctx2_key];
        int p3 = (int)pred3[ctx3_key];
        int h1 = (int)hit1[ctx1_key];
        int h2 = (int)hit2[ctx2_key];
        int h3 = (int)hit3[ctx3_key];
        
        int w1 = 1 + h1;
        int w2 = 2 + h2 * 2;
        int w3 = 3 + h3 * 3;
        int total_w = w1 + w2 + w3;
        int predicted = (p1 * w1 + p2 * w2 + p3 * w3 + total_w / 2) / total_w;
        predicted = predicted & 0xFF;
        
        /* \u0412\u043e\u0441\u0441\u0442\u0430\u043d\u0430\u0432\u043b\u0438\u0432\u0430\u0435\u043c: real = predicted + residual */
        output[i] = (uint8_t)predicted + input[i];
        
        uint8_t actual = output[i];
        if (pred1[ctx1_key] == actual && hit1[ctx1_key] < 255) hit1[ctx1_key]++;
        else { hit1[ctx1_key] = 0; }
        if (pred2[ctx2_key] == actual && hit2[ctx2_key] < 255) hit2[ctx2_key]++;
        else { hit2[ctx2_key] = 0; }
        if (pred3[ctx3_key] == actual && hit3[ctx3_key] < 255) hit3[ctx3_key]++;
        else { hit3[ctx3_key] = 0; }
        
        pred1[ctx1_key] = actual;
        pred2[ctx2_key] = actual;
        pred3[ctx3_key] = actual;
    }
    
    free(pred1); free(pred2); free(pred3);
    free(hit1); free(hit2); free(hit3);
}

/* =====================================================================
 * KOLIBRI FORMULA PREDICTOR — «Number-Thinking» Compression
 * =====================================================================
 * Компактная нейроподобная формула (вдохновлённая KolibriFormula/ResNet)
 * предсказывает байты данных из позиционного контекста.
 *
 * Философия: Данные = Формула + Минимальная коррекция
 *
 * Формула (геном): 48 цифр (0-9), кодирующих 6 «ReasonBlock» слоёв.
 * Каждый слой: [op:1][slope_hi:1][slope_lo:1][bias_hi:1][bias_lo:1]
 *              [aux_hi:1][aux_lo:1][skip_weight:1] = 8 цифр
 *
 * Хранение: 24 байта (packed, 4 бита на цифру) — ультракомпактно!
 * Для блока 65KB: 24 байта формулы vs 65536 байт данных.
 * =====================================================================*/
#define KF_GENE_LEN      64   /* Длина генома (цифры 0-9): 8 слоёв × 8 цифр */
#define KF_GENE_PACKED    32   /* Упакованный размер (байты) */
#define KF_LAYERS          8   /* Количество слоёв-ReasonBlock'ов */

typedef struct {
    uint8_t digits[KF_GENE_LEN];
    double fitness;
} KolibriCompactFormula;

/* --- Контекстно-зависимое предсказание: формула получает предыдущий байт ---
 * Философия: после BWT+MTF, каждый байт сильно зависит от предыдущего.
 * Если prev=0 → скорее всего следующий тоже 0 (серия нулей).
 * Если prev=маленький → скорее всего следующий = 0 (возврат в серию).
 * Формула учится этим переходам за 8 слоёв ReasonBlock'ов.
 */
static int kf_compact_predict(const KolibriCompactFormula *f, int prev_byte) {
    long long x = (long long)(prev_byte & 0xFF);
    
    for (int layer = 0; layer < KF_LAYERS; layer++) {
        int off = layer * 8;
        int op    = f->digits[off] % 6;
        int slope = (int)f->digits[off+1] * 10 + (int)f->digits[off+2] - 50;
        int bias  = (int)f->digits[off+3] * 100 + (int)f->digits[off+4] * 10 
                    + (int)f->digits[off+5] - 500;
        int aux   = (int)f->digits[off+6] * 10 + (int)f->digits[off+7];
        
        long long residual = x;
        long long result;
        
        switch (op) {
            case 0: /* Линейная: slope*x/100 + bias */
                result = (long long)slope * x / 100 + bias; 
                break;
            case 1: /* Пороговая: if |x| < aux then bias, else slope*x/100+bias */
                { long long ax = x < 0 ? -x : x;
                  result = (ax < (long long)aux) ? (long long)bias 
                           : ((long long)slope * x / 100 + bias); }
                break;
            case 2: /* Модульная: (slope*x) % (aux+1) + bias */
                { long long d = (long long)(aux > 0 ? aux : 1);
                  result = ((long long)slope * x / 100) % d + bias; }
                break;
            case 3: /* Мягкая квадратичная: slope*x/(100+|x|) + bias */
                { long long ax = x < 0 ? -x : x;
                  result = (long long)slope * x / (100 + ax) + bias; }
                break;
            case 4: /* Shift: (x >> (aux&3)) + bias */
                result = (x >> ((unsigned)aux & 3)) + bias; 
                break;
            case 5: /* Масштабирование: x * aux / 100 + bias */
                result = x * (long long)aux / 100 + bias;
                break;
            default:
                result = x + bias; 
                break;
        }
        
        /* Клиппинг + Residual Connection */
        if (result > 512LL) result = 512LL;
        if (result < -512LL) result = -512LL;
        x = result + residual;
        if (x > 512LL) x = 512LL;
        if (x < -512LL) x = -512LL;
    }
    
    return (int)(((x % 256) + 256) % 256);
}

/* --- Распаковка генома: 32 байта → 64 цифры --- */
static void kf_unpack_gene(const uint8_t *packed, uint8_t *digits) {
    for (int i = 0; i < KF_GENE_PACKED; i++) {
        digits[i * 2]     = (packed[i] >> 4) & 0x0F;
        digits[i * 2 + 1] = packed[i] & 0x0F;
    }
}

/* --- Секвенциальное декодирование: контекст = предыдущий ДЕКОДИРОВАННЫЙ байт --- */
static void kf_formula_decode_block(const KolibriCompactFormula *formula,
                                     const uint8_t *residuals, size_t len,
                                     uint8_t *output) {
    int prev = 0;
    for (size_t i = 0; i < len; i++) {
        int predicted = kf_compact_predict(formula, prev);
        output[i] = (uint8_t)((int)residuals[i] + predicted);
        prev = (int)output[i]; /* Следующий контекст = текущий ДЕКОДИРОВАННЫЙ байт */
    }
}

#define MATH_BLOCK_SIZE_V40 65536
#define MATH_BLOCK_SIZE    900000  /* v50: 900KB блоки (= bzip2) */

/* ===================== ZRLE (Zero-Run-Length Encoding) v50 ===================== */
/* После BWT+MTF данные содержат множество нулей. ZRLE кодирует серии эффективно:
 * - Ненулевой байт: [byte] (1 байт)
 * - Серия 1..255 нулей:   [0x00] [N-1]       (2 байта)
 * - Серия 256..65535 нулей: [0x00] [0xFF] [hi] [lo] (4 байта)
 * v50: поддержка длинных серий для больших BWT-блоков (512KB+).
 */


static size_t zrle_decode(const uint8_t *input, size_t compressed_len, uint8_t *output, size_t max_out) {
    size_t in_pos = 0, out_pos = 0;
    while (in_pos < compressed_len) {
        if (input[in_pos] == 0) {
            if (in_pos + 1 >= compressed_len) return 0;
            size_t run;
            if (input[in_pos + 1] == 0xFF && in_pos + 3 < compressed_len) {
                /* Длинная серия: [0x00][0xFF][hi][lo], run = hi*256+lo */
                run = ((size_t)input[in_pos + 2] << 8) | input[in_pos + 3];
                if (run == 0) return 0;  /* invalid */
                in_pos += 4;
            } else {
                /* Короткая серия */
                run = (size_t)input[in_pos + 1] + 1;
                in_pos += 2;
            }
            if (out_pos + run > max_out) return 0;
            memset(output + out_pos, 0, run);
            out_pos += run;
        } else {
            if (out_pos >= max_out) return 0;
            output[out_pos++] = input[in_pos++];
        }
    }
    return out_pos;
}
/* Mathematical pattern decompression */
static size_t decompress_mathematical(const uint8_t *input, size_t input_size,
                                     uint8_t *output, size_t output_size,
                                     uint32_t version) {
    
    /* v50: \u043f\u043e\u0434\u0434\u0435\u0440\u0436\u043a\u0430 128KB \u0431\u043b\u043e\u043a\u043e\u0432 (v40 \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u043b 64KB) */
    size_t block_size = (version >= 50) ? MATH_BLOCK_SIZE : MATH_BLOCK_SIZE_V40;
    
    size_t in_pos = 0;
    size_t out_processed = 0;

    /* Generator Persistence */
    KolibriRng current_rng; /* State is maintained across blocks if Method 5 is used */

    while (in_pos < input_size && out_processed < output_size) {
        if (in_pos >= input_size) break;
        
        uint8_t method = input[in_pos++];
        size_t remaining_out = output_size - out_processed;
        size_t block_len = MIN(block_size, remaining_out);
        
        /* If this is the last block, it might be smaller than MATH_BLOCK_SIZE. 
           We know exact output size, so we can clamp.
        */
        
        uint8_t *out_blk = output + out_processed;

        if (method == 0) {
            if (in_pos + block_len > input_size) return 0; 
            memcpy(out_blk, input + in_pos, block_len);
            in_pos += block_len;
        } else if (method == 1) {
            if (in_pos + block_len > input_size) return 0;
            const uint8_t *data = input + in_pos;
            out_blk[0] = data[0];
            for (size_t i = 1; i < block_len; i++) {
                out_blk[i] = out_blk[i-1] + data[i];
            }
            in_pos += block_len;
        } else if (method == 2) {
            if (in_pos + block_len > input_size) return 0;
            const uint8_t *data = input + in_pos; 
            out_blk[0] = data[0];
            out_blk[1] = data[1];
            for (size_t i = 2; i < block_len; i++) {
                out_blk[i] = 2*out_blk[i-1] - out_blk[i-2] + data[i];
            }
            in_pos += block_len;
        } else if (method == 3) {
             if (in_pos + block_len > input_size) return 0;
             const uint8_t *data = input + in_pos;
             memcpy(out_blk, data, 4);
             for (size_t i = 4; i < block_len; i++) {
                 out_blk[i] = out_blk[i-4] + data[i];
             }
             in_pos += block_len;
        } else if (method == 4) {
             /* GENERATOR RECOVERY - INIT */
             if (in_pos + 2 > input_size) {
                 /* Not enough bytes for seed */
                 return 0;
             }
             uint16_t seed = (uint16_t)input[in_pos] | ((uint16_t)input[in_pos+1] << 8);
             in_pos += 2;
             
             k_rng_seed(&current_rng, seed);
             for(size_t i=0; i<block_len; i++) {
                 out_blk[i] = (uint8_t)(k_rng_next(&current_rng) % 256);
             }
        } else if (method == 5) {
             /* GENERATOR RECOVERY - CONTINUE */
             /* Uses current_rng state from previous block */
             for(size_t i=0; i<block_len; i++) {
                 out_blk[i] = (uint8_t)(k_rng_next(&current_rng) % 256);
             }
        } else if (method == 6) {
             /* GENERATOR APPROXIMATION - DELTA */
             if (in_pos + 2 + block_len > input_size) {
                 return 0;
             }
             uint16_t seed = (uint16_t)input[in_pos] | ((uint16_t)input[in_pos+1] << 8);
             in_pos += 2;
             
             KolibriRng rng;
             k_rng_seed(&rng, seed);
             const uint8_t *deltas = input + in_pos;
             
             for(size_t i=0; i<block_len; i++) {
                 uint8_t gen = (uint8_t)(k_rng_next(&rng) % 256);
                 /* Restore: Real = Generator + Delta */
                 out_blk[i] = gen + deltas[i];
             }
             in_pos += block_len;
             
             /* Update continuing seed state for potential next block usage */
             KolibriRng tmp; k_rng_seed(&tmp, seed);
             for(size_t k=0; k<block_len; k++) k_rng_next(&tmp);
             current_rng = tmp;
             
        } else if (method == 8) {
             /* BWT + MTF + ZRLE INVERSE */
             if (in_pos + 7 > input_size) return 0;
             
             uint32_t bwt_primary_d = (uint32_t)input[in_pos]
                                  | ((uint32_t)input[in_pos+1] << 8)
                                  | ((uint32_t)input[in_pos+2] << 16)
                                  | ((uint32_t)input[in_pos+3] << 24);
             in_pos += 4;
             
             /* Читаем размер ZRLE данных */
             uint32_t zrle_size = (uint32_t)input[in_pos]
                               | ((uint32_t)input[in_pos+1] << 8)
                               | ((uint32_t)input[in_pos+2] << 16);
             in_pos += 3;
             
             uint8_t *temp = (uint8_t *)malloc(block_len);
             if (!temp) return 0;
             
             if (zrle_size == 0) {
                 /* Raw MTF данные (ZRLE не использовалась) */
                 if (in_pos + block_len > input_size) { free(temp); return 0; }
                 memcpy(temp, input + in_pos, block_len);
                 in_pos += block_len;
             } else {
                 /* ZRLE декодирование */
                 if (in_pos + zrle_size > input_size) {
                     free(temp); return 0;
                 }
                 size_t decoded = zrle_decode(input + in_pos, zrle_size, temp, block_len);
                 if (decoded != block_len) { free(temp); return 0; }
                 in_pos += zrle_size;
             }
             
             mtf_decode(temp, block_len);
             if (bwt_inverse(temp, block_len, out_blk, bwt_primary_d) != 0) {
                 free(temp);
                 return 0;
             }
             free(temp);
             
        } else if (method == 9) {
             /* CONTEXT PREDICTION + ZRLE INVERSE */
             if (in_pos + 3 > input_size) return 0;
             
             uint32_t zrle_size = (uint32_t)input[in_pos]
                               | ((uint32_t)input[in_pos+1] << 8)
                               | ((uint32_t)input[in_pos+2] << 16);
             in_pos += 3;
             
             uint8_t *temp = (uint8_t *)malloc(block_len);
             if (!temp) return 0;
             
             if (zrle_size == 0) {
                 if (in_pos + block_len > input_size) { free(temp); return 0; }
                 context_predict_decode(input + in_pos, block_len, out_blk);
                 in_pos += block_len;
             } else {
                 if (in_pos + zrle_size > input_size) { free(temp); return 0; }
                 size_t decoded = zrle_decode(input + in_pos, zrle_size, temp, block_len);
                 if (decoded != block_len) { free(temp); return 0; }
                 in_pos += zrle_size;
                 context_predict_decode(temp, block_len, out_blk);
             }
             free(temp);
             
        } else if (method == 10) {
             /* BWT + MTF + CONTEXT + ZRLE INVERSE */
             if (in_pos + 7 > input_size) return 0;
             
             uint32_t bwt_primary_d = (uint32_t)input[in_pos]
                                  | ((uint32_t)input[in_pos+1] << 8)
                                  | ((uint32_t)input[in_pos+2] << 16)
                                  | ((uint32_t)input[in_pos+3] << 24);
             in_pos += 4;
             
             uint32_t zrle_size = (uint32_t)input[in_pos]
                               | ((uint32_t)input[in_pos+1] << 8)
                               | ((uint32_t)input[in_pos+2] << 16);
             in_pos += 3;
             
             uint8_t *temp = (uint8_t *)malloc(block_len);
             if (!temp) return 0;
             
             if (zrle_size == 0) {
                 if (in_pos + block_len > input_size) { free(temp); return 0; }
                 memcpy(temp, input + in_pos, block_len);
                 in_pos += block_len;
             } else {
                 if (in_pos + zrle_size > input_size) { free(temp); return 0; }
                 size_t decoded = zrle_decode(input + in_pos, zrle_size, temp, block_len);
                 if (decoded != block_len) { free(temp); return 0; }
                 in_pos += zrle_size;
             }
             
             /* Inverse Context → MTF → BWT */
             uint8_t *temp2_d = (uint8_t *)malloc(block_len);
             if (!temp2_d) { free(temp); return 0; }
             context_predict_decode(temp, block_len, temp2_d);
             mtf_decode(temp2_d, block_len);
             if (bwt_inverse(temp2_d, block_len, out_blk, bwt_primary_d) != 0) {
                 free(temp); free(temp2_d);
                 return 0;
             }
             free(temp);
             free(temp2_d);
             
        } else if (method == 11) {
             /* BWT + MTF + FORMULA PREDICTION + ZRLE INVERSE
              * Формат: [BWT_primary:4] [gene_packed:32] [zrle_len:3] [zrle_data...] */
             if (in_pos + 4 + KF_GENE_PACKED + 3 > input_size) return 0;
             
             uint32_t bwt_primary_d = (uint32_t)input[in_pos]
                                  | ((uint32_t)input[in_pos+1] << 8)
                                  | ((uint32_t)input[in_pos+2] << 16)
                                  | ((uint32_t)input[in_pos+3] << 24);
             in_pos += 4;
             
             /* Распаковываем геном формулы: 32 байта → 64 цифры */
             KolibriCompactFormula formula_d;
             kf_unpack_gene(input + in_pos, formula_d.digits);
             in_pos += KF_GENE_PACKED;
             
             /* ZRLE размер остатков */
             uint32_t zrle_size = (uint32_t)input[in_pos]
                               | ((uint32_t)input[in_pos+1] << 8)
                               | ((uint32_t)input[in_pos+2] << 16);
             in_pos += 3;
             
             /* Декодируем ZRLE → остатки (residuals) */
             uint8_t *residuals = (uint8_t *)malloc(block_len);
             if (!residuals) return 0;
             
             if (zrle_size == 0) {
                 if (in_pos + block_len > input_size) { free(residuals); return 0; }
                 memcpy(residuals, input + in_pos, block_len);
                 in_pos += block_len;
             } else {
                 if (in_pos + zrle_size > input_size) { free(residuals); return 0; }
                 size_t decoded = zrle_decode(input + in_pos, zrle_size, residuals, block_len);
                 if (decoded != block_len) { free(residuals); return 0; }
                 in_pos += zrle_size;
             }
             
             /* Восстанавливаем BWT+MTF данные: data[i] = predict(prev) + residual[i] */
             uint8_t *mtf_data = (uint8_t *)malloc(block_len);
             if (!mtf_data) { free(residuals); return 0; }
             
             kf_formula_decode_block(&formula_d, residuals, block_len, mtf_data);
             free(residuals);
             
             /* Inverse MTF → BWT */
             mtf_decode(mtf_data, block_len);
             if (bwt_inverse(mtf_data, block_len, out_blk, bwt_primary_d) != 0) {
                 free(mtf_data);
                 return 0;
             }
             free(mtf_data);
             
        } else if (method == 12) {
             /* PURE FORMULA + ZRLE INVERSE (без BWT!)
              * Формат: [gene_packed:32] [zrle_len:3] [zrle_data...] */
             if (in_pos + KF_GENE_PACKED + 3 > input_size) return 0;
             
             KolibriCompactFormula formula_d;
             kf_unpack_gene(input + in_pos, formula_d.digits);
             in_pos += KF_GENE_PACKED;
             
             uint32_t zrle_size = (uint32_t)input[in_pos]
                               | ((uint32_t)input[in_pos+1] << 8)
                               | ((uint32_t)input[in_pos+2] << 16);
             in_pos += 3;
             
             uint8_t *residuals = (uint8_t *)malloc(block_len);
             if (!residuals) return 0;
             
             if (zrle_size == 0) {
                 if (in_pos + block_len > input_size) { free(residuals); return 0; }
                 memcpy(residuals, input + in_pos, block_len);
                 in_pos += block_len;
             } else {
                 if (in_pos + zrle_size > input_size) { free(residuals); return 0; }
                 size_t decoded = zrle_decode(input + in_pos, zrle_size, residuals, block_len);
                 if (decoded != block_len) { free(residuals); return 0; }
                 in_pos += zrle_size;
             }
             
             /* Восстанавливаем данные напрямую: data[i] = predict(prev) + residual[i] */
             kf_formula_decode_block(&formula_d, residuals, block_len, out_blk);
             free(residuals);
             
        } else {
            return 0; /* Unknown method */
        }

        
        out_processed += block_len;
    }

    return out_processed;
}

/* Public API implementation */
KolibriCompressor *kolibri_compressor_create(uint32_t methods) {
    KolibriCompressor *comp = (KolibriCompressor *)calloc(1, sizeof(KolibriCompressor));
    if (!comp) return NULL;

    comp->methods = methods ? methods : KOLIBRI_COMPRESS_ALL;
    comp->temp_buffer = NULL;
    comp->temp_buffer_size = 0;

    return comp;
}

void kolibri_compressor_destroy(KolibriCompressor *comp) {
    if (!comp) return;
    free(comp->temp_buffer);
    free(comp);
}


/* ====================================================================
 * TOKEN-LEVEL TEXT STREAM (v52)
 * ====================================================================
 * Простая токенизация текста:
 *   - Слова (A-Za-z0-9_) кодируются как токены по словарю
 *   - Остальные байты идут как есть
 *   - Escape 0xFE используется для меток токенов
 * Формат токенов в потоке:
 *   0xFE 0x00           -> literal 0xFE
 *   0xFE 0x01 <id>      -> token dictionary word
 */
#define TOKEN_ESCAPE 0xFE
#define TOKEN_LITERAL 0x00
#define TOKEN_DICT_MARK 0x01

#define TOKEN_DICT_MAX 256
#define TOKEN_COLLECT_MAX 2048
#define TOKEN_HASH_SIZE 4096
#define TOKEN_MIN_LEN 2
#define TOKEN_MAX_LEN 32

typedef struct {
    uint32_t hash;
    uint16_t len;
    uint32_t count;
    uint32_t str_off;
} TokenEntry;

typedef struct {
    TokenEntry entries[TOKEN_COLLECT_MAX];
    int32_t table[TOKEN_HASH_SIZE];
    uint32_t entry_count;
    uint8_t *blob;
    uint32_t blob_size;
    uint32_t blob_cap;
} TokenCollector;

typedef struct {
    uint16_t count;
    uint16_t len[TOKEN_DICT_MAX];
    uint32_t off[TOKEN_DICT_MAX];
    uint8_t *blob;
    uint32_t blob_size;
} TokenDict;

typedef struct {
    uint32_t idx;
    uint32_t score;
} TokenScore;

static int token_score_cmp(const void *a, const void *b) {
    const TokenScore *x = (const TokenScore *)a;
    const TokenScore *y = (const TokenScore *)b;
    if (x->score < y->score) return 1;
    if (x->score > y->score) return -1;
    return 0;
}

static inline int token_is_word(uint8_t c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_';
}

static inline uint32_t token_hash(const uint8_t *s, uint16_t len) {
    uint32_t h = 2166136261u;
    for (uint16_t i = 0; i < len; i++) h = (h ^ s[i]) * 16777619u;
    return h;
}

static void token_collector_init(TokenCollector *tc, size_t input_size) {
    tc->entry_count = 0;
    for (size_t i = 0; i < TOKEN_HASH_SIZE; i++) tc->table[i] = -1;
    tc->blob_cap = (uint32_t)MIN(input_size, (size_t)(1024 * 1024));
    tc->blob_size = 0;
    tc->blob = (uint8_t *)malloc(tc->blob_cap);
}

static void token_collector_free(TokenCollector *tc) {
    free(tc->blob);
    tc->blob = NULL;
}

static void token_collector_add(TokenCollector *tc, const uint8_t *word, uint16_t len) {
    if (len < TOKEN_MIN_LEN || len > TOKEN_MAX_LEN) return;
    if (tc->entry_count >= TOKEN_COLLECT_MAX || !tc->blob) return;
    uint32_t h = token_hash(word, len);
    uint32_t idx = h & (TOKEN_HASH_SIZE - 1);
    for (uint32_t step = 0; step < TOKEN_HASH_SIZE; step++) {
        int32_t slot = tc->table[idx];
        if (slot < 0) {
            if (tc->blob_size + len > tc->blob_cap) return;
            TokenEntry *e = &tc->entries[tc->entry_count];
            e->hash = h;
            e->len = len;
            e->count = 1;
            e->str_off = tc->blob_size;
            memcpy(tc->blob + tc->blob_size, word, len);
            tc->blob_size += len;
            tc->table[idx] = (int32_t)tc->entry_count;
            tc->entry_count++;
            return;
        }
        TokenEntry *e = &tc->entries[(uint32_t)slot];
        if (e->hash == h && e->len == len &&
            memcmp(tc->blob + e->str_off, word, len) == 0) {
            e->count++;
            return;
        }
        idx = (idx + 1) & (TOKEN_HASH_SIZE - 1);
    }
}

static int token_dict_build(TokenDict *dict, const uint8_t *input, size_t input_size) {
    TokenCollector tc;
    token_collector_init(&tc, input_size);
    if (!tc.blob) return 0;

    size_t i = 0;
    while (i < input_size) {
        if (token_is_word(input[i])) {
            size_t start = i;
            size_t len = 0;
            while (i < input_size && token_is_word(input[i]) && len < TOKEN_MAX_LEN) {
                i++;
                len++;
            }
            token_collector_add(&tc, input + start, (uint16_t)len);
        } else {
            i++;
        }
    }

    TokenScore *scores = (TokenScore *)malloc(tc.entry_count * sizeof(TokenScore));
    if (!scores) { token_collector_free(&tc); return 0; }

    uint32_t sc_count = 0;
    for (uint32_t k = 0; k < tc.entry_count; k++) {
        TokenEntry *e = &tc.entries[k];
        if (e->count >= 2 && e->len >= TOKEN_MIN_LEN && e->len <= TOKEN_MAX_LEN) {
            scores[sc_count].idx = k;
            scores[sc_count].score = e->count * e->len;
            sc_count++;
        }
    }

    if (sc_count == 0) {
        free(scores);
        token_collector_free(&tc);
        return 0;
    }

    qsort(scores, sc_count, sizeof(TokenScore), token_score_cmp);

    dict->count = 0;
    dict->blob_size = 0;
    dict->blob = (uint8_t *)malloc(tc.blob_size);
    if (!dict->blob) {
        free(scores);
        token_collector_free(&tc);
        return 0;
    }

    for (uint32_t k = 0; k < sc_count && dict->count < TOKEN_DICT_MAX; k++) {
        TokenEntry *e = &tc.entries[scores[k].idx];
        dict->len[dict->count] = e->len;
        dict->off[dict->count] = dict->blob_size;
        memcpy(dict->blob + dict->blob_size, tc.blob + e->str_off, e->len);
        dict->blob_size += e->len;
        dict->count++;
    }

    free(scores);
    token_collector_free(&tc);
    return dict->count > 0;
}


static void token_dict_free(TokenDict *dict) {
    free(dict->blob);
    dict->blob = NULL;
    dict->blob_size = 0;
    dict->count = 0;
}

static size_t token_dict_write(const TokenDict *dict, uint8_t *output, size_t output_max) {
    size_t need = 2;
    for (uint16_t i = 0; i < dict->count; i++) need += 1 + dict->len[i];
    if (need > output_max) return 0;
    output[0] = (uint8_t)(dict->count & 0xFF);
    output[1] = (uint8_t)((dict->count >> 8) & 0xFF);
    size_t pos = 2;
    for (uint16_t i = 0; i < dict->count; i++) {
        output[pos++] = (uint8_t)dict->len[i];
        memcpy(output + pos, dict->blob + dict->off[i], dict->len[i]);
        pos += dict->len[i];
    }
    return pos;
}

static int token_dict_read(const uint8_t *input, size_t input_size,
                           size_t *consumed, TokenDict *dict) {
    if (input_size < 2) return 0;
    uint16_t count = (uint16_t)input[0] | ((uint16_t)input[1] << 8);
    if (count > TOKEN_DICT_MAX) return 0;
    size_t pos = 2;
    uint32_t total = 0;
    for (uint16_t i = 0; i < count; i++) {
        if (pos >= input_size) return 0;
        uint8_t len = input[pos++];
        if (pos + len > input_size) return 0;
        total += len;
        pos += len;
    }
    dict->blob = (uint8_t *)malloc(total);
    if (!dict->blob) return 0;
    dict->count = count;
    dict->blob_size = total;
    pos = 2;
    uint32_t off = 0;
    for (uint16_t i = 0; i < count; i++) {
        uint8_t len = input[pos++];
        dict->len[i] = len;
        dict->off[i] = off;
        memcpy(dict->blob + off, input + pos, len);
        pos += len;
        off += len;
    }
    *consumed = pos;
    return 1;
}

static int token_dict_find(const TokenDict *dict, const uint8_t *word,
                           uint16_t len, uint8_t *out_id) {
    for (uint16_t i = 0; i < dict->count; i++) {
        if (dict->len[i] == len &&
            memcmp(dict->blob + dict->off[i], word, len) == 0) {
            *out_id = (uint8_t)i;
            return 1;
        }
    }
    return 0;
}

static size_t token_encode_text(const uint8_t *input, size_t input_size,
                                uint8_t *output, size_t output_max,
                                const TokenDict *dict) {
    size_t ip = 0;
    size_t op = 0;
    while (ip < input_size) {
        if (token_is_word(input[ip])) {
            size_t start = ip;
            size_t len = 0;
            while (ip < input_size && token_is_word(input[ip]) && len < 255) {
                ip++;
                len++;
            }
            uint8_t id = 0;
            if (len >= TOKEN_MIN_LEN && token_dict_find(dict, input + start, (uint16_t)len, &id)) {
                if (op + 3 > output_max) return 0;
                output[op++] = TOKEN_ESCAPE;
                output[op++] = TOKEN_DICT_MARK;
                output[op++] = id;
            } else {
                for (size_t k = 0; k < len; k++) {
                    uint8_t c = input[start + k];
                    if (c == TOKEN_ESCAPE) {
                        if (op + 2 > output_max) return 0;
                        output[op++] = TOKEN_ESCAPE;
                        output[op++] = TOKEN_LITERAL;
                    } else {
                        if (op + 1 > output_max) return 0;
                        output[op++] = c;
                    }
                }
            }
        } else {
            uint8_t c = input[ip++];
            if (c == TOKEN_ESCAPE) {
                if (op + 2 > output_max) return 0;
                output[op++] = TOKEN_ESCAPE;
                output[op++] = TOKEN_LITERAL;
            } else {
                if (op + 1 > output_max) return 0;
                output[op++] = c;
            }
        }
    }
    return op;
}

static size_t token_decode_text(const uint8_t *input, size_t input_size,
                                uint8_t *output, size_t output_max,
                                const TokenDict *dict, size_t expected_size) {
    size_t ip = 0;
    size_t op = 0;
    while (ip < input_size && op < expected_size) {
        uint8_t c = input[ip++];
        if (c != TOKEN_ESCAPE) {
            if (op >= output_max) return 0;
            output[op++] = c;
            continue;
        }
        if (ip >= input_size) return 0;
        uint8_t tag = input[ip++];
        if (tag == TOKEN_LITERAL) {
            if (op >= output_max) return 0;
            output[op++] = TOKEN_ESCAPE;
        } else if (tag == TOKEN_DICT_MARK) {
            if (ip >= input_size) return 0;
            uint8_t id = input[ip++];
            if (id >= dict->count) return 0;
            uint16_t len = dict->len[id];
            if (op + len > output_max) return 0;
            memcpy(output + op, dict->blob + dict->off[id], len);
            op += len;
        } else {
            return 0;
        }
    }
    return (op == expected_size) ? op : 0;
}


/* ====================================================================
 * LZ-LITE: быстрый предпроход для удаления длинных повторов
 * ====================================================================
 * Окно 32KB, хеш-цепочки, макс. совпадение 258 байт.
 * Формат: литерал <0xFE, byte> или <len_hi|0x80, len_lo, dist_hi, dist_lo>
 *   literal: если byte != 0xFE → один байт
 *            если byte == 0xFE → <0xFE, 0xFE>
 *   match:   <0xFF, len-3, dist_hi, dist_lo>  (len 3..258, dist 1..32768)
 * ==================================================================== */

#define LZ_WBITS   20           /* v55: 1 MB окно */
#define LZ_WSIZE   (1 << LZ_WBITS)  /* 1048576 */
#define LZ_WMASK   (LZ_WSIZE - 1)
#define LZ_HTBITS  20           /* v55: 1M хеш-таблица */
#define LZ_HTSIZE  (1 << LZ_HTBITS)
#define LZ_HTMASK  (LZ_HTSIZE - 1)
#define LZ_MIN_MATCH  3
#define LZ_MAX_MATCH  259       /* v55: extended len-4 max 255 */
#define LZ_MAX_CHAIN  512       /* макс. глубина цепочки (v53+) */
#define LZ_NICE_MATCH 128       /* v59: ранний выход при хорошем совпадении */
#define LZ_ESCAPE     0xFF      /* escape-байт для совпадений */
#define LZ_EXT_CODE   0xFC      /* v55: extended 24-bit distance match */
#define LZ_REP_CODE   0xFD      /* v54: опкод rep-match (LZMA-style) */
#define LZ_NUM_REPS   4         /* кол-во хранимых последних дистанций */

static inline uint32_t lz_hash4(const uint8_t *p) {
    uint32_t h;
    memcpy(&h, p, 4); /* v58: быстрое чтение (unaligned OK на x86/arm) */
    return (h * 0x9E3779B1u) >> (32 - LZ_HTBITS);
}

/* Кодирование LZ-lite: input→output, возвращает размер или 0 при неудаче */
static size_t lz_lite_encode(const uint8_t *input, size_t input_size,
                              uint8_t *output, size_t output_max)
{
    if (input_size < 4) return 0;

    int32_t *head = (int32_t *)calloc(LZ_HTSIZE, sizeof(int32_t));
    int32_t *prev = (int32_t *)calloc(LZ_WSIZE, sizeof(int32_t));
    if (!head || !prev) { free(head); free(prev); return 0; }

    /* v54: Rep-match — последние 4 дистанции (LZMA-style) */
    int rep[LZ_NUM_REPS] = {0, 0, 0, 0};

    /* -1 = нет ссылки */
    for (size_t j = 0; j < (size_t)LZ_HTSIZE; j++) head[j] = -1;
    for (size_t j = 0; j < (size_t)LZ_WSIZE; j++) prev[j] = -1;

    size_t op = 0;
    size_t ip = 0;
    size_t limit = input_size - 3; /* нужно 4 байта для хеша */

    /* v54: Near-Optimal LZ parsing helper — поиск лучшего матча от позиции p */
    #define LZ_FIND_MATCH(pos_, best_l_, best_d_)                          \
    do {                                                                    \
        best_l_ = 0; best_d_ = 0;                                          \
        if ((pos_) < limit) {                                               \
            uint32_t hh_ = lz_hash4(input + (pos_));                        \
            int32_t cc_ = head[hh_];                                         \
            int ch_ = 0;                                                     \
            while (cc_ >= 0 && ch_ < 256) {  /* v60: near-optimal chain/2 */ \
                size_t ca_ = (size_t)cc_;                                    \
                if ((pos_) > ca_ && ((pos_) - ca_) <= LZ_WSIZE) {           \
                    const uint8_t *aa_ = input + (pos_);                     \
                    const uint8_t *bb_ = input + ca_;                        \
                    int ll_ = 0;                                             \
                    int mx_ = (int)MIN((size_t)LZ_MAX_MATCH, input_size-(pos_)); \
                    /* v61: 8-байтное сравнение */                             \
                    if (mx_ >= 4) {                                          \
                        uint32_t va_, vb_;                                   \
                        memcpy(&va_, aa_, 4); memcpy(&vb_, bb_, 4);          \
                        if (va_ == vb_) {                                    \
                            ll_ = 4;                                         \
                            while (ll_ + 8 <= mx_) {                         \
                                uint64_t va8_,vb8_;                          \
                                memcpy(&va8_, aa_+ll_, 8);                   \
                                memcpy(&vb8_, bb_+ll_, 8);                   \
                                if (va8_ != vb8_) break;                     \
                                ll_ += 8;                                    \
                            }                                                \
                            while (ll_ < mx_ && aa_[ll_] == bb_[ll_]) ll_++; \
                        }                                                    \
                    } else {                                                 \
                        while (ll_ < mx_ && aa_[ll_] == bb_[ll_]) ll_++;     \
                    }                                                        \
                    if (ll_ >= LZ_MIN_MATCH && ll_ > best_l_) {              \
                        best_l_ = ll_; best_d_ = (int)((pos_) - ca_);       \
                        if (ll_ >= LZ_NICE_MATCH) break;                     \
                    }                                                        \
                }                                                            \
                cc_ = prev[ca_ & LZ_WMASK];                                 \
                if (cc_ >= (int32_t)(pos_)) break;                           \
                if (cc_ >= 0) __builtin_prefetch(input+(size_t)cc_, 0, 0);   \
                ch_++;                                                       \
            }                                                                \
        }                                                                    \
    } while(0)

    while (ip < input_size) {
        int best_len = 0;
        int best_dist = 0;
        int best_rep = -1;  /* v54: индекс rep-match (-1 = обычный матч) */

        /* v54: Rep-match поиск — проверяем последние 4 дистанции (LZMA-style)
         * Rep-match кодируется за 3 байта (вместо 3-4 для обычного),
         * поэтому выгоднее при прочих равных условиях */
        if (ip < limit) {
            int max_rep_len = (int)MIN((size_t)LZ_MAX_MATCH, input_size - ip);
            for (int ri = 0; ri < LZ_NUM_REPS; ri++) {
                if (rep[ri] <= 0 || rep[ri] > (int)ip) continue;
                const uint8_t *a = input + ip;
                const uint8_t *b = input + ip - rep[ri];
                int rlen = 0;
                /* v61: fast 4-byte check + 8-byte inner loop */
                if (max_rep_len >= 4) {
                    uint32_t rv, rbv;
                    memcpy(&rv, a, 4); memcpy(&rbv, b, 4);
                    if (rv == rbv) {
                        rlen = 4;
                        while (rlen + 8 <= max_rep_len) {
                            uint64_t rv8, rbv8;
                            memcpy(&rv8, a+rlen, 8); memcpy(&rbv8, b+rlen, 8);
                            if (rv8 != rbv8) break;
                            rlen += 8;
                        }
                        while (rlen < max_rep_len && a[rlen] == b[rlen]) rlen++;
                    }
                } else {
                    while (rlen < max_rep_len && a[rlen] == b[rlen]) rlen++;
                }
                /* Rep-match дешевле: 3B vs 3-4B, бонус к длине */
                if (rlen >= LZ_MIN_MATCH && rlen > best_len) {
                    best_len = rlen;
                    best_dist = rep[ri];
                    best_rep = ri;
                }
            }
        }

        /* Обычный хеш-цепочный поиск (если rep не дал NICE_MATCH) */
        if (ip < limit && best_len < LZ_NICE_MATCH) {
            uint32_t h = lz_hash4(input + ip);
            int32_t cur = head[h];
            int chain = 0;

            while (cur >= 0 && chain < LZ_MAX_CHAIN) {
                size_t candidate = (size_t)cur;
                if (ip > candidate && (ip - candidate) <= LZ_WSIZE) {
                    const uint8_t *a = input + ip;
                    const uint8_t *b = input + candidate;
                    int len = 0;
                    int max_possible = (int)MIN((size_t)LZ_MAX_MATCH, input_size - ip);
                    /* v61: 8-байтное сравнение */
                    if (max_possible >= 4) {
                        uint32_t va, vb;
                        memcpy(&va, a, 4); memcpy(&vb, b, 4);
                        if (va == vb) {
                            len = 4;
                            while (len + 8 <= max_possible) {
                                uint64_t va8, vb8;
                                memcpy(&va8, a+len, 8); memcpy(&vb8, b+len, 8);
                                if (va8 != vb8) break;
                                len += 8;
                            }
                            while (len < max_possible && a[len] == b[len]) len++;
                        }
                    } else {
                        while (len < max_possible && a[len] == b[len]) len++;
                    }

                    /* Обычный матч должен быть длиннее rep-match на 1+,
                     * т.к. rep-match на 1 байт дешевле в кодировании */
                    if (len >= LZ_MIN_MATCH && len > best_len + (best_rep >= 0 ? 1 : 0)) {
                        best_len = len;
                        best_dist = (int)(ip - candidate);
                        best_rep = -1;  /* обычный матч */
                        if (len >= LZ_NICE_MATCH) break;
                    }
                }
                cur = prev[candidate & LZ_WMASK];
                if (cur >= (int32_t)ip) break;
                if (cur >= 0) __builtin_prefetch(input + (size_t)cur, 0, 0);
                chain++;
                if (best_len >= 32 && chain >= 64) break; /* v60: adaptive chain */
            }

            /* Обновляем хеш-цепочку */
            prev[ip & LZ_WMASK] = head[h];
            head[h] = (int32_t)ip;
        } else if (ip < limit) {
            /* Если rep-match уже NICE — всё равно обновляем хеш */
            uint32_t h = lz_hash4(input + ip);
            prev[ip & LZ_WMASK] = head[h];
            head[h] = (int32_t)ip;
        }

        if (best_len >= LZ_MIN_MATCH) {
            /* v54 Near-Optimal: проверяем ip+1 и ip+2 (если не rep-NICE) */
            if (best_len < LZ_NICE_MATCH && best_rep < 0) {
                int len1 = 0, dist1 = 0;
                if ((ip + 1) < limit) {
                    LZ_FIND_MATCH(ip + 1, len1, dist1);
                }
                int len2 = 0, dist2 = 0;
                if ((ip + 2) < limit) {
                    LZ_FIND_MATCH(ip + 2, len2, dist2);
                }

                /* v55: стоимость учитывает 24-bit extended (6B) */
                int mc0 = (best_len==3 && best_dist<=256) ? 3 : (best_dist>65535 ? 6 : 4);
                int mc1 = (len1==3 && dist1<=256) ? 3 : (dist1>65535 ? 6 : 4);
                int mc2 = (len2==3 && dist2<=256) ? 3 : (dist2>65535 ? 6 : 4);
                int cost0 = best_len - mc0;
                int cost1 = len1 - mc1
                            - (input[ip] == LZ_ESCAPE ? 2 : 1);
                int cost2 = len2 - mc2
                            - (input[ip] == LZ_ESCAPE ? 2 : 1)
                            - (input[ip+1] == LZ_ESCAPE ? 2 : 1);

                if (cost2 > cost0 && cost2 > cost1 && len2 >= LZ_MIN_MATCH) {
                    for (int s = 0; s < 2; s++) {
                        if (input[ip] == LZ_ESCAPE) {
                            if (op + 2 > output_max) goto fail;
                            output[op++] = LZ_ESCAPE; output[op++] = LZ_ESCAPE;
                        } else {
                            if (op + 1 > output_max) goto fail;
                            output[op++] = input[ip];
                        }
                        prev[ip & LZ_WMASK] = head[lz_hash4(input + ip)];
                        head[lz_hash4(input + ip)] = (int32_t)ip;
                        ip++;
                    }
                    best_len = len2; best_dist = dist2;
                    best_rep = -1;
                } else if (cost1 > cost0 && len1 >= LZ_MIN_MATCH) {
                    if (input[ip] == LZ_ESCAPE) {
                        if (op + 2 > output_max) goto fail;
                        output[op++] = LZ_ESCAPE; output[op++] = LZ_ESCAPE;
                    } else {
                        if (op + 1 > output_max) goto fail;
                        output[op++] = input[ip];
                    }
                    prev[ip & LZ_WMASK] = head[lz_hash4(input + ip)];
                    head[lz_hash4(input + ip)] = (int32_t)ip;
                    ip++;
                    best_len = len1; best_dist = dist1;
                    best_rep = -1;
                }
            }

            /* v55 Match encoding:
             * Rep:      0xFF 0xFD (rep_idx<<6|len-3)         (3B, len 3-66)
             * Short:    0xFF 0xFE dist_lo                    (3B, len=3, dist≤256)
             * Long:     0xFF (len-4) dist_hi dist_lo         (4B, len 4-255, dist≤65535)
             * Extended: 0xFF 0xFC (len-4) d[23:16] d[15:8] d[7:0]  (6B, 24-bit dist) */
            int rep_max_len = 66;
            /* FIX: если rep-match слишком длинный для rep-формата,
             * кодируем как обычный — сбрасываем best_rep для sync rep MRU */
            if (best_rep >= 0 && best_len > rep_max_len) {
                best_rep = -1;
            }
            if (best_rep >= 0 && best_len <= rep_max_len) {
                int rl = best_len;
                if (rl > rep_max_len) rl = rep_max_len;
                if (op + 3 > output_max) goto fail;
                output[op++] = LZ_ESCAPE;
                output[op++] = LZ_REP_CODE;
                output[op++] = (uint8_t)((best_rep << 6) | (rl - 3));
                best_len = rl;
            } else if (best_len == 3 && best_dist <= 256) {
                if (op + 3 > output_max) goto fail;
                output[op++] = LZ_ESCAPE;
                output[op++] = 0xFE;
                output[op++] = (uint8_t)(best_dist - 1);
            } else if (best_dist > 65535) {
                /* v55: Extended 24-bit distance match */
                int el = best_len;
                if (el > 259) el = 259;
                if (op + 6 > output_max) goto fail;
                output[op++] = LZ_ESCAPE;
                output[op++] = LZ_EXT_CODE;
                output[op++] = (uint8_t)(el - 4);
                output[op++] = (uint8_t)((best_dist >> 16) & 0xFF);
                output[op++] = (uint8_t)((best_dist >> 8) & 0xFF);
                output[op++] = (uint8_t)(best_dist & 0xFF);
                best_len = el;
            } else {
                int rl = best_len;
                if (rl > 255) rl = 255; /* regular long max len-4=0xFB=251 */
                if (op + 4 > output_max) goto fail;
                output[op++] = LZ_ESCAPE;
                output[op++] = (uint8_t)(rl - 4);
                output[op++] = (uint8_t)((best_dist >> 8) & 0xFF);
                output[op++] = (uint8_t)(best_dist & 0xFF);
                best_len = rl;
            }

            /* v54: обновляем rep-дистанции (MRU — последний вперёд) */
            if (best_rep < 0) {
                /* Новая дистанция — сдвигаем всё вправо */
                for (int ri = LZ_NUM_REPS - 1; ri > 0; ri--)
                    rep[ri] = rep[ri - 1];
                rep[0] = best_dist;
            } else if (best_rep > 0) {
                /* rep[1..3] → двигаем на позицию 0 (MRU) */
                int tmp = rep[best_rep];
                for (int ri = best_rep; ri > 0; ri--)
                    rep[ri] = rep[ri - 1];
                rep[0] = tmp;
            }
            /* best_rep == 0: rep[0] уже актуален, ничего не меняем */

            /* v60: sparse hash update — для длинных матчей обновляем каждую 4-ю */
            for (int k = 1; k < best_len && (ip + k) < limit; k++) {
                if (k > 32 && (k & 3)) continue;
                uint32_t hk = lz_hash4(input + ip + k);
                prev[(ip + k) & LZ_WMASK] = head[hk];
                head[hk] = (int32_t)(ip + k);
            }
            ip += best_len;
        } else {
            /* Литерал */
            if (input[ip] == LZ_ESCAPE) {
                if (op + 2 > output_max) goto fail;
                output[op++] = LZ_ESCAPE;
                output[op++] = LZ_ESCAPE;
            } else {
                if (op + 1 > output_max) goto fail;
                output[op++] = input[ip];
            }
            ip++;
        }
    }

    free(head);
    free(prev);

    /* Выгодно только если уменьшили */
    if (op >= input_size) return 0;
    return op;

fail:
    free(head);
    free(prev);
    return 0;
}

/* ====================================================================
 * LZ-ULTRAFAST v75: минимальная латентность — 16K хеш (64KB memset)
 * ====================================================================
 * Формат выхода совместим с LZ-lite (lz_lite_decode без изменений).
 * v75b: 16K хеш (64KB stack) вместо 64K (256KB) — быстрее memset.
 * Целевая скорость: ~500 MB/s (~0.2 мс на 100KB).
 * ==================================================================== */
#define ULTRAFAST_HBITS  14
#define ULTRAFAST_HSIZE  (1u << ULTRAFAST_HBITS)
#define ULTRAFAST_HMASK  (ULTRAFAST_HSIZE - 1u)

static inline uint32_t uf_hash4(const uint8_t *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return (v * 0x9E3779B1u) >> (32 - ULTRAFAST_HBITS);
}

static size_t lz_ultrafast_encode(const uint8_t *input, size_t input_size,
                                   uint8_t *output, size_t output_max)
{
    if (input_size < 8) return 0;

    /* Stack-allocated хеш-таблица 16K × 4B = 64KB */
    int32_t head[ULTRAFAST_HSIZE];
    memset(head, -1, sizeof(head));

    size_t op = 0, ip = 0;
    const size_t limit = input_size - 4;
    /* Safe zone: прекращаем без bounds check пока op < safe_end */
    const size_t safe_end = (output_max > 16) ? output_max - 16 : 0;

    while (ip <= limit && op < safe_end) {
        uint32_t h = uf_hash4(input + ip);
        int32_t cur = head[h];
        head[h] = (int32_t)ip;

        if (cur >= 0 && ip > (size_t)cur && (ip - (size_t)cur) <= 65535) {
            const uint8_t *a = input + ip;
            const uint8_t *b = input + (size_t)cur;
            int mx = (int)((input_size - ip < 259) ? input_size - ip : 259);
            uint32_t va, vb;
            memcpy(&va, a, 4); memcpy(&vb, b, 4);
            if (va == vb) {
                int len = 4;
                while (len + 8 <= mx) {
                    uint64_t va8, vb8;
                    memcpy(&va8, a + len, 8);
                    memcpy(&vb8, b + len, 8);
                    if (va8 != vb8) break;
                    len += 8;
                }
                while (len < mx && a[len] == b[len]) len++;
                if (len >= LZ_MIN_MATCH) {
                    int dist = (int)(ip - (size_t)cur);
                    if (len == 3 && dist <= 256) {
                        output[op++] = LZ_ESCAPE;
                        output[op++] = 0xFE;
                        output[op++] = (uint8_t)(dist - 1);
                    } else {
                        if (len > 255) len = 255;
                        output[op++] = LZ_ESCAPE;
                        output[op++] = (uint8_t)(len - 4);
                        output[op++] = (uint8_t)((dist >> 8) & 0xFF);
                        output[op++] = (uint8_t)(dist & 0xFF);
                    }
                    /* Минимальный hash update: только середину матча */
                    if (len > 4 && (ip + (len >> 1)) <= limit) {
                        size_t mid = ip + (size_t)(len >> 1);
                        head[uf_hash4(input + mid)] = (int32_t)mid;
                    }
                    ip += len;
                    continue;
                }
            }
        }

        /* Литерал */
        if (input[ip] == LZ_ESCAPE) {
            output[op++] = LZ_ESCAPE; output[op++] = LZ_ESCAPE;
        } else {
            output[op++] = input[ip];
        }
        ip++;
    }

    /* Хвост: побайтово с проверкой bounds */
    while (ip < input_size) {
        if (input[ip] == LZ_ESCAPE) {
            if (op + 2 > output_max) return 0;
            output[op++] = LZ_ESCAPE; output[op++] = LZ_ESCAPE;
        } else {
            if (op + 1 > output_max) return 0;
            output[op++] = input[ip];
        }
        ip++;
    }

    if (op >= input_size) return 0;
    return op;
}

/* ====================================================================
 * LZ-BLAZING v84: ультра-скорость — таргет <1ms на 500KB
 * ====================================================================
 * Стратегия v84: минимальная стоимость на одну позицию ввода:
 *   1. hash4: 1 умножение, fast golden-ratio hash
 *   2. 16K хеш-таблица (64KB) — хорошо сидит в L2 кеше
 *   3. Нулевой hash update внутри матчей: экономим random writes
 *   4. LZ4-style acceleration: step=1+(miss>>5) на несжимаемых участках
 *   5. 16-байтный SSE2 bulk literal: проверка+копия 16B за раз
 *   6. Rep-match: повторная дистанция без хеш-lookup (LZMA-style speedup)
 *   - Формат 100% совместим с lz_lite_decode (без изменений)
 * ==================================================================== */
#define BLAZING_HBITS  14
#define BLAZING_HSIZE  (1u << BLAZING_HBITS)
#define BLAZING_HMASK  (BLAZING_HSIZE - 1u)

static inline uint32_t blazing_hash4(const uint8_t *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return (v * 0x9E3779B1u) >> (32 - BLAZING_HBITS);
}

/* Проверка: есть ли байт LZ_ESCAPE (0xFF) в 8-байтном слове (SWAR bit-trick) */
static inline int blazing_has_escape8(uint64_t v) {
    uint64_t x = v ^ 0xFFFFFFFFFFFFFFFFULL;  /* ~v: ищем нулевые байты = 0xFF в оригинале */
    return (int)(((x - 0x0101010101010101ULL) & ~x & 0x8080808080808080ULL) != 0);
}

static size_t lz_blazing_v76_encode(const uint8_t *restrict input, size_t input_size,
                                     uint8_t *restrict output, size_t output_max)
{
    if (input_size < 8) return 0;

    /* 16K × 4B = 64KB — хорошо сидит в L2, быстрый memset */
    int32_t head[BLAZING_HSIZE];
    memset(head, -1, sizeof(head));

    size_t op = 0, ip = 0;
    const size_t limit = input_size - 4;
    const size_t safe_end = (output_max > 16) ? output_max - 16 : 0;
    int miss_count = 0;
    int last_dist = 0;  /* v84: Rep-match — последняя использованная дистанция */

    /* v84: SSE2 вектор для поиска escape-байта 0xFF в 16-байтных чанках */
#if KF_USE_SIMD
    const __m128i esc_vec = _mm_set1_epi8((char)0xFF);
#endif

    while (ip <= limit && op < safe_end) {

        /* === v84: Rep-match — проверяем последнюю дистанцию ДО хеш-lookup ===
         * Дешёвая проверка: 1 байт → 4 байта → расширение.
         * Если попадает — экономим хеш + таблицу (≈6ns/позицию). */
        if (last_dist > 0 && ip >= (size_t)last_dist) {
            if (input[ip] == input[ip - last_dist]) {
                uint32_t va, vb;
                memcpy(&va, input + ip, 4);
                memcpy(&vb, input + ip - last_dist, 4);
                if (va == vb) {
                    const uint8_t *a = input + ip;
                    const uint8_t *b = input + ip - last_dist;
                    int mx = (int)((input_size - ip < 259) ? input_size - ip : 259);
                    int len = 4;
                    while (len + 8 <= mx) {
                        uint64_t qa, qb;
                        memcpy(&qa, a + len, 8);
                        memcpy(&qb, b + len, 8);
                        if (qa != qb) break;
                        len += 8;
                    }
                    while (len < mx && a[len] == b[len]) len++;
                    /* Обновляем хеш для текущей позиции */
                    head[blazing_hash4(input + ip)] = (int32_t)ip;
                    if (len > 255) len = 255;
                    output[op++] = LZ_ESCAPE;
                    output[op++] = (uint8_t)(len - 4);
                    output[op++] = (uint8_t)((last_dist >> 8) & 0xFF);
                    output[op++] = (uint8_t)(last_dist & 0xFF);
                    ip += len;
                    miss_count = 0;
                    continue;
                }
            }
        }

        uint32_t h = blazing_hash4(input + ip);
        int32_t cur = head[h];
        head[h] = (int32_t)ip;

        if (cur >= 0 && ip > (size_t)cur && (ip - (size_t)cur) <= 65535) {
            const uint8_t *a = input + ip;
            const uint8_t *b = input + (size_t)cur;
            uint32_t va, vb;
            memcpy(&va, a, 4); memcpy(&vb, b, 4);
            if (va == vb) {
                int mx = (int)((input_size - ip < 259) ? input_size - ip : 259);
                int len = 4;
                while (len + 8 <= mx) {
                    uint64_t qa, qb;
                    memcpy(&qa, a + len, 8);
                    memcpy(&qb, b + len, 8);
                    if (qa != qb) break;
                    len += 8;
                }
                while (len < mx && a[len] == b[len]) len++;
                if (len >= LZ_MIN_MATCH) {
                    int dist = (int)(ip - (size_t)cur);
                    last_dist = dist;  /* v84: запоминаем дистанцию для rep-match */
                    if (len == 3 && dist <= 256) {
                        output[op++] = LZ_ESCAPE;
                        output[op++] = 0xFE;
                        output[op++] = (uint8_t)(dist - 1);
                    } else {
                        if (len > 255) len = 255;
                        output[op++] = LZ_ESCAPE;
                        output[op++] = (uint8_t)(len - 4);
                        output[op++] = (uint8_t)((dist >> 8) & 0xFF);
                        output[op++] = (uint8_t)(dist & 0xFF);
                    }
                    ip += len;
                    miss_count = 0;
                    continue;
                }
            }
        }

        /* v84: Литерал: 16-byte SSE2 bulk path — проверка на 0xFF через _mm_cmpeq_epi8 */
#if KF_USE_SIMD
        if (ip + 16 <= limit && op + 16 <= safe_end) {
            __m128i chunk = _mm_loadu_si128((const __m128i *)(input + ip));
            int esc_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(chunk, esc_vec));
            if (esc_mask == 0) {
                /* Нет escape (0xFF) в 16 байтах — SSE2 bulk copy */
                _mm_storeu_si128((__m128i *)(output + op), chunk);
                op += 16;
                ip += 16;
                miss_count += 16;
                continue;
            }
        }
#endif
        /* Fallback: 8-byte SWAR bulk path */
        if (ip + 8 <= limit && op + 8 <= safe_end) {
            uint64_t v8;
            memcpy(&v8, input + ip, 8);
            if (!blazing_has_escape8(v8)) {
                memcpy(output + op, input + ip, 8);
                op += 8;
                ip += 8;
                miss_count += 8;
                continue;
            }
        }
        /* Slow path: побайтовый вывод литерала */
        if (input[ip] == LZ_ESCAPE) {
            output[op++] = LZ_ESCAPE; output[op++] = LZ_ESCAPE;
        } else {
            output[op++] = input[ip];
        }
        miss_count++;
        {
            int skip = 1 + (miss_count >> 6);  /* LZ4-style ускорение */
            /* Выводим пропущенные позиции (при skip>1) как литералы */
            for (int s = 1; s < skip && (ip + s) < input_size && op < safe_end; s++) {
                if (input[ip + s] == LZ_ESCAPE) {
                    output[op++] = LZ_ESCAPE; output[op++] = LZ_ESCAPE;
                } else {
                    output[op++] = input[ip + s];
                }
            }
            ip += skip;
        }
    }

    /* Хвост */
    while (ip < input_size) {
        if (input[ip] == LZ_ESCAPE) {
            if (op + 2 > output_max) return 0;
            output[op++] = LZ_ESCAPE; output[op++] = LZ_ESCAPE;
        } else {
            if (op + 1 > output_max) return 0;
            output[op++] = input[ip];
        }
        ip++;
    }

    if (op >= input_size) return 0;
    return op;
}

/* ====================================================================
 * LZ-TURBO v63: максимальная скорость — 8-step chains + greedy
 * ====================================================================
 * Формат выхода совместим с LZ-lite (lz_lite_decode без изменений).
 * Отличия от LZ-lite:
 *   - Макс. глубина цепочки 8 (vs 512) — ~64× меньше сравнений
 *   - Greedy (vs near-optimal ip+1/ip+2 проверки)
 *   - Без rep-match (экономим bookkeeping)
 *   - Обновление хеша каждые 3 байта внутри матча (vs каждый)
 * Pipeline: input → turbo LZ → output (без CM/RC/Formula)
 * Скорость: ~100-200 MB/s vs ~4 MB/s у v62 = 25-50× быстрее
 * ==================================================================== */
#define TURBO_MAX_CHAIN 8  /* макс. глубина цепочки (vs 512 у lz_lite) */

static size_t lz_turbo_encode(const uint8_t *input, size_t input_size,
                               uint8_t *output, size_t output_max)
{
    if (input_size < 8) return 0;

    int32_t *head = (int32_t *)malloc(LZ_HTSIZE * sizeof(int32_t));
    int32_t *prev = (int32_t *)malloc(LZ_WSIZE * sizeof(int32_t));
    if (!head || !prev) { free(head); free(prev); return 0; }
    for (size_t j = 0; j < (size_t)LZ_HTSIZE; j++) head[j] = -1;
    for (size_t j = 0; j < (size_t)LZ_WSIZE; j++) prev[j] = -1;

    size_t op = 0, ip = 0;
    size_t limit = input_size - 3;

    while (ip < input_size) {
        int best_len = 0, best_dist = 0;

        if (ip < limit) {
            uint32_t h = lz_hash4(input + ip);
            int32_t cur = head[h];
            int chain = 0;

            while (cur >= 0 && chain < TURBO_MAX_CHAIN) {
                size_t cand = (size_t)cur;
                if (ip > cand && (ip - cand) <= LZ_WSIZE) {
                    const uint8_t *a = input + ip;
                    const uint8_t *b = input + cand;
                    int len = 0;
                    int mx = (int)MIN((size_t)LZ_MAX_MATCH, input_size - ip);
                    if (mx >= 4) {
                        uint32_t va, vb;
                        memcpy(&va, a, 4); memcpy(&vb, b, 4);
                        if (va == vb) {
                            len = 4;
                            while (len + 8 <= mx) {
                                uint64_t va8, vb8;
                                memcpy(&va8, a + len, 8);
                                memcpy(&vb8, b + len, 8);
                                if (va8 != vb8) break;
                                len += 8;
                            }
                            while (len < mx && a[len] == b[len]) len++;
                        }
                    } else {
                        while (len < mx && a[len] == b[len]) len++;
                    }
                    if (len >= LZ_MIN_MATCH && len > best_len) {
                        best_len = len;
                        best_dist = (int)(ip - cand);
                        if (len >= 64) break; /* достаточно хороший матч */
                    }
                }
                cur = prev[cand & LZ_WMASK];
                if (cur >= (int32_t)ip) break;
                chain++;
            }

            /* Обновляем цепочку */
            prev[ip & LZ_WMASK] = head[h];
            head[h] = (int32_t)ip;
        }

        if (best_len >= LZ_MIN_MATCH) {
            /* Кодируем матч в формате LZ-lite */
            if (best_len == 3 && best_dist <= 256) {
                if (op + 3 > output_max) goto turbo_fail;
                output[op++] = LZ_ESCAPE;
                output[op++] = 0xFE;
                output[op++] = (uint8_t)(best_dist - 1);
            } else if (best_dist > 65535) {
                if (best_len > 259) best_len = 259;
                if (op + 6 > output_max) goto turbo_fail;
                output[op++] = LZ_ESCAPE;
                output[op++] = LZ_EXT_CODE;
                output[op++] = (uint8_t)(best_len - 4);
                output[op++] = (uint8_t)((best_dist >> 16) & 0xFF);
                output[op++] = (uint8_t)((best_dist >> 8) & 0xFF);
                output[op++] = (uint8_t)(best_dist & 0xFF);
            } else {
                if (best_len > 255) best_len = 255;
                if (op + 4 > output_max) goto turbo_fail;
                output[op++] = LZ_ESCAPE;
                output[op++] = (uint8_t)(best_len - 4);
                output[op++] = (uint8_t)((best_dist >> 8) & 0xFF);
                output[op++] = (uint8_t)(best_dist & 0xFF);
            }
            /* Обновляем хеш для позиций внутри матча (каждые 3-й для скорости) */
            for (int k = 1; k < best_len && (ip + k) < limit; k += 3) {
                uint32_t hk = lz_hash4(input + ip + k);
                prev[(ip + k) & LZ_WMASK] = head[hk];
                head[hk] = (int32_t)(ip + k);
            }
            ip += best_len;
        } else {
            /* Литерал */
            if (input[ip] == LZ_ESCAPE) {
                if (op + 2 > output_max) goto turbo_fail;
                output[op++] = LZ_ESCAPE; output[op++] = LZ_ESCAPE;
            } else {
                if (op + 1 > output_max) goto turbo_fail;
                output[op++] = input[ip];
            }
            ip++;
        }
    }

    free(head); free(prev);
    if (op >= input_size) return 0;
    return op;

turbo_fail:
    free(head); free(prev);
    return 0;
}

/* Декодирование LZ-lite */
static size_t lz_lite_decode(const uint8_t *input, size_t input_size,
                              uint8_t *output, size_t output_max,
                              size_t original_size)
{
    size_t ip = 0, op = 0;

    /* v54: Rep-match — последние 4 дистанции */
    int rep[LZ_NUM_REPS] = {0, 0, 0, 0};

    while (ip < input_size && op < original_size) {
        uint8_t c = input[ip++];
        if (c == LZ_ESCAPE) {
            if (ip >= input_size) return 0;
            uint8_t next = input[ip++];
            if (next == LZ_ESCAPE) {
                /* Экранированный литерал 0xFF */
                if (op >= output_max) return 0;
                output[op++] = LZ_ESCAPE;
            } else if (next == LZ_REP_CODE) {
                /* v54: Rep-match: 0xFF 0xFD (rep_idx<<6 | len-3) */
                if (ip >= input_size) return 0;
                uint8_t rb = input[ip++];
                int rep_idx = (rb >> 6) & 3;
                int len = (rb & 0x3F) + 3;
                int dist = rep[rep_idx];
                if (dist == 0 || dist > (int)op) return 0;
                if (op + len > output_max) return 0;
                for (int k = 0; k < len; k++) {
                    output[op] = output[op - dist];
                    op++;
                }
                /* MRU обновление rep */
                if (rep_idx > 0) {
                    int tmp = rep[rep_idx];
                    for (int ri = rep_idx; ri > 0; ri--)
                        rep[ri] = rep[ri - 1];
                    rep[0] = tmp;
                }
            } else if (next == LZ_EXT_CODE) {
                /* v55: Extended 24-bit distance match */
                if (ip + 3 >= input_size) return 0;
                int len = (int)input[ip] + 4;
                int dist = ((int)input[ip+1] << 16)
                         | ((int)input[ip+2] << 8)
                         | (int)input[ip+3];
                ip += 4;
                if (dist == 0 || dist > (int)op) return 0;
                if (op + len > output_max) return 0;
                for (int k = 0; k < len; k++) {
                    output[op] = output[op - dist];
                    op++;
                }
                for (int ri = LZ_NUM_REPS - 1; ri > 0; ri--)
                    rep[ri] = rep[ri - 1];
                rep[0] = dist;
            } else if (next == 0xFE) {
                /* Short match: len=3, dist=next_byte+1 */
                if (ip >= input_size) return 0;
                int len = 3;
                int dist = (int)input[ip] + 1;
                ip += 1;
                if (dist == 0 || dist > (int)op) return 0;
                if (op + len > output_max) return 0;
                for (int k = 0; k < len; k++) {
                    output[op] = output[op - dist];
                    op++;
                }
                for (int ri = LZ_NUM_REPS - 1; ri > 0; ri--)
                    rep[ri] = rep[ri - 1];
                rep[0] = dist;
            } else {
                /* Long match: len = next+4, dist = 2 bytes */
                if (ip + 1 >= input_size) return 0;
                int len = (int)next + 4;
                int dist = ((int)input[ip] << 8) | (int)input[ip + 1];
                ip += 2;
                if (dist == 0 || dist > (int)op) return 0;
                if (op + len > output_max) return 0;
                for (int k = 0; k < len; k++) {
                    output[op] = output[op - dist];
                    op++;
                }
                for (int ri = LZ_NUM_REPS - 1; ri > 0; ri--)
                    rep[ri] = rep[ri - 1];
                rep[0] = dist;
            }
        } else {
            if (op >= output_max) return 0;
            output[op++] = c;
        }
    }

    return op;
}

/* ====================================================================
 * KOLIBRI FORMULA COMPRESSION v51 — "Kolibri Mind"
 * ====================================================================
 * Чистое формульное сжатие: адаптивное предсказание + range coding.
 *
 * Архитектура:
 *   - Range Coder (fpaq0-style, 32-bit)
 *   - Раздельные таблицы для каждого порядка контекста (Order 0..7)
 *   - SSE (Secondary Symbol Estimation)
 *   - Память: ~4.2 MB  (раздельные таблицы, нет коллизий между порядками)
 *   - Никакого BWT, LZ77, Huffman — чистая формула
 * ==================================================================== */

/* --- Range Coder --- */
typedef struct {
    uint32_t x1, x2, x;
    uint8_t *buf;
    size_t pos, size;
} KolibriRC;

static void krc_enc_init(KolibriRC *c, uint8_t *buf, size_t sz) {
    c->x1 = 0; c->x2 = 0xFFFFFFFFu; c->x = 0;
    c->buf = buf; c->pos = 0; c->size = sz;
}
static inline void krc_enc_bit(KolibriRC *c, int bit, uint32_t prob) {
    uint32_t xmid = c->x1 + ((c->x2 - c->x1) >> 12) * prob;
    if (bit) c->x2 = xmid; else c->x1 = xmid + 1;
    while (((c->x1 ^ c->x2) & 0xFF000000u) == 0) {
        if (c->pos < c->size) c->buf[c->pos++] = (uint8_t)(c->x1 >> 24);
        c->x1 <<= 8; c->x2 = (c->x2 << 8) | 0xFF;
    }
}
static void krc_enc_flush(KolibriRC *c) {
    for (int i = 0; i < 4; i++) {
        if (c->pos < c->size) c->buf[c->pos++] = (uint8_t)(c->x1 >> 24);
        c->x1 <<= 8;
    }
}
static void krc_dec_init(KolibriRC *c, const uint8_t *buf, size_t sz) {
    c->x1 = 0; c->x2 = 0xFFFFFFFFu;
    c->buf = (uint8_t *)buf; c->pos = 0; c->size = sz;
    c->x = 0;
    for (int i = 0; i < 4; i++)
        c->x = (c->x << 8) | (c->pos < c->size ? c->buf[c->pos++] : 0);
}
static inline int krc_dec_bit(KolibriRC *c, uint32_t prob) {
    uint32_t xmid = c->x1 + ((c->x2 - c->x1) >> 12) * prob;
    int bit = (c->x <= xmid) ? 1 : 0;
    if (bit) c->x2 = xmid; else c->x1 = xmid + 1;
    while (((c->x1 ^ c->x2) & 0xFF000000u) == 0) {
        c->x1 <<= 8; c->x2 = (c->x2 << 8) | 0xFF;
        c->x = (c->x << 8) | (c->pos < c->size ? c->buf[c->pos++] : 0);
    }
    return bit;
}

/* --- Контекстная модель с раздельными таблицами ---
 *
 * Размеры таблиц (степени двойки для быстрого маскирования):
 *   O0:   256 × 256 = 65536 (128KB) — полная статистика (без хеша)
 *   O1: 64K записей  (128KB)
 *   O2: 128K записей (256KB)
 *   O3: 256K записей (512KB)
 *   O4: 256K записей (512KB)
 *   O5: 1M записей (2MB)
 *   O6: 1M записей (2MB)
 *   O7: 1M записей (2MB)
 *   O8: 1M записей (2MB) — v55
 *   SSE: 256×8×33    (135KB)
 *   Run: 4K записей (8KB) — v55
 *   Итого: ~11.4 MB
 */
#define T0_SIZE  65536u     /* Order-0: 256 byte contexts × 256 bit-tree */
#define T1_SIZE  65536u     /* 64K */
#define T2_SIZE  262144u    /* v85: 256K — optimal for 471K data */
#define T3_SIZE  1048576u   /* 1M */
#define T4_SIZE  2097152u   /* 2M */
#define T5_SIZE  2097152u   /* 2M */
#define T6_SIZE  2097152u   /* 2M */
#define T7_SIZE  1048576u   /* 1M */
#define T8_SIZE  1048576u   /* 1M */
#define SSE_Q    33         /* v73: 33 бина — оптимальное квантование */
#define SSE_SZ   (512u * 8u * SSE_Q)
#define TRUN_SIZE  16384u    /* v70: run-length context — perfectly sized (63*256+255=16383) */
#define APM_SIZE   32768u   /* v59: state-aware APM — (cx*2+is_lz)*33+q2 */

/* v66: восстановлены раздельные O6/O7/O8 таблицы (были слиты в v62, теряли точность) */
#define KF62_NUM_PREDS 15       /* v66: 15 предикторов (O0-O8 раздельно + state/word/sparse/run + match + nibble) */
#define KF62_PAD       16       /* v62: padding до 16 для SIMD (2 × __m128i) */

/* v65: APM2 (3-й уровень адаптивного квантования вероятностей) */
#define APM2_SIZE  (512u * 33u)   /* v73: 33 бина — лучшая конвергенция на малых данных */
#define KF62_BLOCK_SIZE 8388608  /* v69: блок 8MB — максимум данных для обучения CM */

/* v66: Match Model — увеличенная таблица */
/* v71: 2-way set-associative hash — 2 позиции на слот, уменьшает коллизии вдвое */
#define MATCH_HT_BITS  21
#define MATCH_HT_SIZE  (1u << MATCH_HT_BITS)
#define MATCH_HT_MASK  (MATCH_HT_SIZE - 1)
#define MATCH_HT_WAYS  2    /* v71: 2-way set-associative */

/* v66: Nibble model — предсказание по верхним 4 битам текущего байта */
#define TNIBBLE_SIZE   131072u  /* v67: 128K — больше контекстов для nibble */

/* v67: Character Class model — предсказание по типу символов */
#define TCC_SIZE       16384u   /* 16K: (cc0*4+cc1)*256+cx */

/* v71: Literal-After-Match adaptive model — расширенная таблица
 * LZMA-style: после match(dist=d), предсказываем литерал по input[pos-d].
 * Индекс: match_byte * 256 + cx (65536 entries — полный байт вместо nibble) */
#define TLAM_SIZE      65536u   /* v71: 256 match_bytes × 256 cx */

/* --- v54: Логистическое смешивание (stretch/squash) ---
 * stretch(p) = ln(p/(1-p)) — переводит вероятность в logit-пространство
 * squash(x) = 1/(1+exp(-x)) — обратно в вероятность
 * Таблицы предвычислены для 12-бит вероятностей (0..4095) */
static int16_t stretch_table[4096];
static uint16_t squash_table[4096]; /* индекс: x+2048 → вероятность 0..4095 */
static int kf_tables_ready = 0;

static void kf_init_tables(void) {
    if (kf_tables_ready) return;
    /* stretch: p в 0..4095 → logit в ~[-2048..2047] */
    for (int i = 1; i < 4095; i++) {
        double p = (double)i / 4096.0;
        double logit = log(p / (1.0 - p)) * 256.0; /* масштаб ×256 */
        if (logit < -2047) logit = -2047;
        if (logit > 2047) logit = 2047;
        stretch_table[i] = (int16_t)logit;
    }
    stretch_table[0] = -2047;
    stretch_table[4095] = 2047;
    /* squash: x в -2048..2047 → p в 0..4095 */
    for (int i = 0; i < 4096; i++) {
        int x = i - 2048;
        double ex = exp(-(double)x / 256.0);
        double p = 1.0 / (1.0 + ex);
        int q = (int)(p * 4096.0 + 0.5);
        if (q < 1) q = 1; if (q > 4095) q = 4095;
        squash_table[i] = (uint16_t)q;
    }
    kf_tables_ready = 1;
}

/* ============================================================================
 * v70: Move-to-Front (MTF) преобразование
 * ============================================================================
 * Классическое преобразование из bzip2: после BWT данные содержат длинные
 * серии одинаковых символов. MTF конвертирует каждый символ в его позицию
 * в динамически обновляемой таблице. Повторяющиеся символы → 0,
 * недавно виденные → малые числа. CM модель предсказывает это идеально.
 * ============================================================================ */
static void kolibri_mtf_forward(uint8_t *data, size_t len) {
    uint8_t table[256] __attribute__((aligned(16)));
    for (int i = 0; i < 256; i++) table[i] = (uint8_t)i;
    for (size_t i = 0; i < len; i++) {
        uint8_t c = data[i];
        int j = 0;
#if KF_USE_SIMD
        /* v71: SIMD MTF — 16-параллельный поиск через _mm_cmpeq_epi8 */
        __m128i needle = _mm_set1_epi8((char)c);
        for (; j + 16 <= 256; j += 16) {
            __m128i chunk = _mm_load_si128((const __m128i*)(table + j));
            __m128i eq = _mm_cmpeq_epi8(chunk, needle);
            int mask = _mm_movemask_epi8(eq);
            if (mask) {
                j += __builtin_ctz((unsigned)mask);
                goto found;
            }
        }
        while (table[j] != c) j++;
found:
#else
        while (table[j] != c) j++;
#endif
        data[i] = (uint8_t)j;
        /* Сдвигаем элементы, помещаем c в начало */
        if (j > 0) {
            memmove(table + 1, table, (size_t)j);
            table[0] = c;
        }
    }
}

static void kolibri_mtf_inverse(uint8_t *data, size_t len) {
    uint8_t table[256] __attribute__((aligned(16)));
    for (int i = 0; i < 256; i++) table[i] = (uint8_t)i;
    for (size_t i = 0; i < len; i++) {
        int j = (int)data[i];
        uint8_t c = table[j];
        data[i] = c;
        if (j > 0) {
            memmove(table + 1, table, (size_t)j);
            table[0] = c;
        }
    }
}

/* v70: Delta кодирование — разница между соседними байтами.
 * BWT выход имеет плавные переходы → дельта конвертирует в малые значения
 * вокруг 0, которые CM предсказывает значительно лучше. */
static void kolibri_delta_forward(uint8_t *data, size_t len) {
    if (len < 2) return;
    uint8_t prev = data[0];
    for (size_t i = len - 1; i >= 1; i--) {
        data[i] = (uint8_t)(data[i] - data[i - 1]);
    }
}

static void kolibri_delta_inverse(uint8_t *data, size_t len) {
    if (len < 2) return;
    for (size_t i = 1; i < len; i++) {
        data[i] = (uint8_t)(data[i] + data[i - 1]);
    }
}

/* ============================================================================
 * v77: Zero-RLE кодирование для MTF потока
 * ============================================================================
 * После BWT+MTF ~55% байт = 0 (MTF rank 0). Длинные серии нулей плохо
 * сжимаются побитовым CM (0.5-1 бит/байт). RLE кодирует их компактно.
 *
 * Формат (escape=0xFF, крайне редок после MTF):
 *   values 0-254: pass through (1 byte)
 *   literal 255:  [0xFF, 0x00] (2 bytes — escape + flag "not a run")
 *   run of N≥3 zeros: [0xFF, N-1] where N-1 = 2..255 (runs 3..256)
 *   singles/pairs of zeros: pass through (1-2 bytes as-is)
 * ============================================================================ */
static size_t kolibri_zero_rle_encode(const uint8_t *input, size_t len,
                                       uint8_t *output, size_t out_max) {
    size_t ip = 0, op = 0;
    while (ip < len) {
        if (input[ip] == 0) {
            /* Считаем длину серии нулей */
            size_t count = 0;
            size_t save = ip;
            while (ip < len && input[ip] == 0 && count < 256) {
                count++;
                ip++;
            }
            if (count >= 2) {
                /* v80: Серия 2+ нулей: [0xFF, count-1] — помогает CM */
                if (op + 2 > out_max) return 0;
                output[op++] = 0xFF;
                output[op++] = (uint8_t)(count - 1);
            } else {
                /* Одиночный ноль: pass through */
                if (op + count > out_max) return 0;
                for (size_t j = 0; j < count; j++) output[op++] = 0;
            }
        } else if (input[ip] == 0xFF) {
            /* Escape literal 255: [0xFF, 0x00] */
            if (op + 2 > out_max) return 0;
            output[op++] = 0xFF;
            output[op++] = 0x00;
            ip++;
        } else {
            if (op + 1 > out_max) return 0;
            output[op++] = input[ip++];
        }
    }
    return op;
}

static size_t kolibri_zero_rle_decode(const uint8_t *input, size_t len,
                                       uint8_t *output, size_t out_max) {
    size_t ip = 0, op = 0;
    while (ip < len) {
        if (input[ip] == 0xFF) {
            if (ip + 1 >= len) return 0; /* truncated */
            uint8_t code = input[ip + 1];
            if (code == 0x00) {
                /* Literal 255 */
                if (op >= out_max) return 0;
                output[op++] = 0xFF;
            } else {
                /* Run of (code+1) zeros, code ≥ 2 → run ≥ 3 */
                size_t count = (size_t)code + 1;
                if (op + count > out_max) return 0;
                memset(output + op, 0, count);
                op += count;
            }
            ip += 2;
        } else {
            if (op >= out_max) return 0;
            output[op++] = input[ip++];
        }
    }
    return op;
}

static inline int16_t kf_stretch(uint32_t p) {
    if (p > 4095) p = 4095;
    return stretch_table[p];
}
static inline uint32_t kf_squash(int32_t x) {
    int idx = x + 2048;
    if (idx < 0) idx = 0;
    if (idx > 4095) idx = 4095;
    return squash_table[idx];
}

/* v57: Модель: 13 предикторов + APM chain + per-bit weights
 * O0-O8, LZ state, word boundary, sparse, run.
 * APM (2-й SSE) уточняет вероятность по контексту partial byte (cx). */

#define KF_NUM_PREDS 13  /* кол-во предикторов */

typedef struct {
    uint16_t *t0, *t1, *t2, *t3, *t4, *t5, *t6, *t7;
    uint16_t *t8;        /* v55: Order-8 context */
    uint16_t *sse;
    uint16_t *apm;       /* v56: 2nd SSE (APM chain) */
    uint16_t *tstate;    /* LZ state-aware */
    uint16_t *tword;     /* word boundary */
    uint16_t *tsparse;   /* sparse context */
    uint16_t *trun;      /* v55: run-length context */
    /* v60: per-bit weights (8 позиций × 13 предикторов) */
    int32_t w[8][KF_NUM_PREDS];
    int32_t wsum;
} KF51M;

#define TSTATE_SIZE  524288u /* v67: 512K — больше контекстов для state machine */
#define TWORD_SIZE   262144u        /* v68: 256K word boundary context — меньше коллизий */
#define TSPARSE_SIZE 262144u        /* v68: 256K sparse skip-gram context */

static uint16_t *kf51_new(size_t n) {
    uint16_t *p = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (p) for (size_t i = 0; i < n; i++) p[i] = 2048;
    return p;
}
static int kf51_init(KF51M *m) {
    kf_init_tables();
    m->t0 = kf51_new(T0_SIZE);  m->t1 = kf51_new(T1_SIZE);
    m->t2 = kf51_new(T2_SIZE);  m->t3 = kf51_new(T3_SIZE);
    m->t4 = kf51_new(T4_SIZE);  m->t5 = kf51_new(T5_SIZE);
    m->t6 = kf51_new(T6_SIZE);  m->t7 = kf51_new(T7_SIZE);
    m->t8 = kf51_new(T8_SIZE);
    m->sse = kf51_new(SSE_SZ);
    m->apm = kf51_new(APM_SIZE);    /* v56: 2nd SSE */
    m->tstate = kf51_new(TSTATE_SIZE);
    m->tword = kf51_new(TWORD_SIZE);
    m->tsparse = kf51_new(TSPARSE_SIZE);
    m->trun = kf51_new(TRUN_SIZE);
    /* v60: per-bit weights — отдельные веса для каждой битовой позиции */
    for (int bp = 0; bp < 8; bp++) {
        m->w[bp][0]  =  4;  m->w[bp][1]  =  4;  m->w[bp][2]  =  8;
        m->w[bp][3]  = 16;  m->w[bp][4]  = 32;  m->w[bp][5]  = 48;
        m->w[bp][6]  = 72;  m->w[bp][7]  = 112; m->w[bp][8]  = 128;
        m->w[bp][9]  = 32;  m->w[bp][10] = 12;  m->w[bp][11] = 12;
        m->w[bp][12] = 16;
    }
    m->wsum = 0;
    return (m->t0 && m->t1 && m->t2 && m->t3 &&
            m->t4 && m->t5 && m->t6 && m->t7 && m->t8 &&
            m->sse && m->apm && m->tstate && m->tword &&
            m->tsparse && m->trun);
}
static void kf51_destroy(KF51M *m) {
    free(m->t0); free(m->t1); free(m->t2); free(m->t3);
    free(m->t4); free(m->t5); free(m->t6); free(m->t7);
    free(m->t8); free(m->sse); free(m->apm); free(m->tstate);
    free(m->tword); free(m->tsparse); free(m->trun);
}

/* =====================================================================
 * v66 MODEL: раздельные O6/O7/O8, Nibble, улучшенный Match, SIMD
 * Память: ~15.2 MB — больше контекстов = лучше предсказание
 * ===================================================================== */
typedef struct {
    uint16_t *t0, *t1, *t2, *t3, *t4, *t5;
    uint16_t *t6, *t7, *t8;  /* v66: раздельные O6/O7/O8 (были слиты в v62) */
    uint16_t *sse, *apm;
    uint16_t *tstate, *tword, *tsparse, *trun;
    uint16_t *tnibble;       /* v66: Nibble model — верхний полубайт */
    uint16_t *tcc;            /* v67: Character Class model */
    /* v70: LZMA-style literal-after-match adaptive table */
    uint16_t *tlam;           /* indexed by (match_byte_high << 4 | cx_bits) */
    /* v64: Match Model — предсказание по длинным совпадениям */
    /* v71: 2-way set-associative — 2 позиции на слот */
    uint32_t *match_ht;       /* хеш-таблица: 4-byte context → позиция (×2 ways) */
    uint8_t  *match_buf;      /* буфер обработанных байт */
    size_t    match_buf_pos;  /* текущая позиция записи */
    size_t    match_pos;      /* позиция совпадения */
    int       match_len;      /* длина текущего совпадения */
    int       match_active;   /* есть ли активное совпадение */
    uint8_t   match_byte;     /* предсказанный байт из совпадения */
    /* v79: RLE-aware state machine — 2 состояния вместо 10 для BWT+RLE данных */
    uint8_t   rle_mode;        /* 0=LZCM state machine, 1=RLE 2-state machine */
    /* v65: APM2 — 3-й уровень адаптивного квантования */
    uint16_t *apm2;           /* 3-я ступень SSE-цепочки */
    /* v62: int16 веса, aligned для SSE2 */
    /* v70: 16 наборов — 8 битовых позиций × 2 (alpha vs non-alpha) */
    int16_t w[32][KF62_PAD] __attribute__((aligned(16)));
} KF62M;

static int kf62_init(KF62M *m) {
    kf_init_tables();
    m->t0     = kf51_new(T0_SIZE);   m->t1    = kf51_new(T1_SIZE);
    m->t2     = kf51_new(T2_SIZE);   m->t3    = kf51_new(T3_SIZE);
    m->t4     = kf51_new(T4_SIZE);   m->t5    = kf51_new(T5_SIZE);
    /* v66: раздельные O6/O7/O8 — 3 таблицы вместо одной слитой */
    m->t6     = kf51_new(T6_SIZE);
    m->t7     = kf51_new(T7_SIZE);
    m->t8     = kf51_new(T8_SIZE);
    m->sse    = kf51_new(SSE_SZ);
    m->apm    = kf51_new(APM_SIZE);
    m->tstate = kf51_new(TSTATE_SIZE);
    m->tword  = kf51_new(TWORD_SIZE);
    m->tsparse= kf51_new(TSPARSE_SIZE);
    m->trun   = kf51_new(TRUN_SIZE);
    /* v66: Nibble model */
    m->tnibble= kf51_new(TNIBBLE_SIZE);
    /* v67: Character Class model */
    m->tcc    = kf51_new(TCC_SIZE);
    /* v70: Literal-After-Match adaptive model */
    m->tlam   = kf51_new(TLAM_SIZE);
    /* v71: Match Model — 2-way set-associative (2M × 2 ways = 4M entries) */
    m->match_ht  = (uint32_t *)calloc(MATCH_HT_SIZE * MATCH_HT_WAYS, sizeof(uint32_t));
    m->match_buf = (uint8_t *)calloc(KF62_BLOCK_SIZE + 256, 1);
    m->match_buf_pos = 0;
    m->match_pos = 0;
    m->match_len = 0;
    m->match_active = 0;
    m->match_byte = 0;
    m->rle_mode = 0;  /* v79: по умолчанию LZCM 10-state machine */
    /* v65: APM2 — 3-я ступень адаптивной цепочки */
    m->apm2 = kf51_new(APM2_SIZE);
    /* v70: начальные веса для 15 предикторов × 16 наборов (8 bp × 2 alpha) */
    for (int bp = 0; bp < 32; bp++) {
        m->w[bp][0]  =  2;   /* O0 */
        m->w[bp][1]  =  4;   /* O1 */
        m->w[bp][2]  =  8;   /* O2 */
        m->w[bp][3]  = 16;   /* O3 */
        m->w[bp][4]  = 32;   /* O4 */
        m->w[bp][5]  = 48;   /* O5 */
        m->w[bp][6]  = 64;   /* O6 */
        m->w[bp][7]  = 80;   /* O7 */
        m->w[bp][8]  = 96;   /* O8 — сильный, длинный контекст */
        m->w[bp][9]  = 32;   /* state */
        m->w[bp][10] = 16;   /* word */
        m->w[bp][11] = 12;   /* sparse */
        m->w[bp][12] = 16;   /* run */
        m->w[bp][13] = 24;   /* match */
        m->w[bp][14] = 8;    /* nibble */
        m->w[bp][15] = 12;   /* v67: char class */
    }
    return (m->t0 && m->t1 && m->t2 && m->t3 && m->t4 && m->t5 &&
            m->t6 && m->t7 && m->t8 &&
            m->sse && m->apm && m->tstate &&
            m->tword && m->tsparse && m->trun && m->tnibble && m->tcc &&
            m->tlam && m->match_ht && m->match_buf && m->apm2);
}

/* v82: Fast init — без match hash table (экономия 16 MB для BWT пути).
 * match_ht=NULL → KF62_PROCESS_BYTE пропускает match model,
 * s_[13]=0 всегда. Для BWT+RLE данных match почти бесполезен,
 * но его 16 MB таблица выталкивает полезные данные из L3 cache. */
static int kf62_init_fast(KF62M *m) {
    kf_init_tables();
    m->t0     = kf51_new(T0_SIZE);   m->t1    = kf51_new(T1_SIZE);
    m->t2     = kf51_new(T2_SIZE);   m->t3    = kf51_new(T3_SIZE);
    m->t4     = kf51_new(T4_SIZE);   m->t5    = kf51_new(T5_SIZE);
    m->t6     = kf51_new(T6_SIZE);
    m->t7     = kf51_new(T7_SIZE);
    m->t8     = kf51_new(T8_SIZE);
    m->sse    = kf51_new(SSE_SZ);
    m->apm    = kf51_new(APM_SIZE);
    m->tstate = kf51_new(TSTATE_SIZE);
    m->tword  = kf51_new(TWORD_SIZE);
    m->tsparse= kf51_new(TSPARSE_SIZE);
    m->trun   = kf51_new(TRUN_SIZE);
    m->tnibble= kf51_new(TNIBBLE_SIZE);
    m->tcc    = kf51_new(TCC_SIZE);
    m->tlam   = kf51_new(TLAM_SIZE);
    /* v82: БЕЗ match_ht / match_buf — s_[13]=0 всегда */
    m->match_ht  = NULL;
    m->match_buf = NULL;
    m->match_buf_pos = 0;
    m->match_pos = 0;
    m->match_len = 0;
    m->match_active = 0;
    m->match_byte = 0;
    m->rle_mode = 0;  /* v84: tested rle_mode=1 — хуже на 354 байт, оставляем 0 */
    m->apm2 = kf51_new(APM2_SIZE);
    for (int bp = 0; bp < 32; bp++) {
        m->w[bp][0]  =  2;   m->w[bp][1]  =  4;
        m->w[bp][2]  =  8;   m->w[bp][3]  = 16;
        m->w[bp][4]  = 32;   m->w[bp][5]  = 48;
        m->w[bp][6]  = 64;   m->w[bp][7]  = 80;
        m->w[bp][8]  = 96;   m->w[bp][9]  = 32;
        m->w[bp][10] = 16;   m->w[bp][11] = 12;
        m->w[bp][12] = 16;   m->w[bp][13] =  0;  /* match weight — match_ht=NULL → always 0 */
        m->w[bp][14] =  8;   m->w[bp][15] = 12;
    }
    return (m->t0 && m->t1 && m->t2 && m->t3 &&
            m->t4 && m->t5 &&
            m->t6 && m->t7 && m->t8 &&
            m->sse && m->apm && m->tstate &&
            m->tword && m->tsparse && m->trun && m->tnibble && m->tcc &&
            m->tlam && m->apm2);
}

static void kf62_destroy(KF62M *m) {
    free(m->t0);  free(m->t1);  free(m->t2);  free(m->t3);
    free(m->t4);  free(m->t5);
    free(m->t6);  free(m->t7);  free(m->t8);  /* v66: раздельные O6/O7/O8 */
    free(m->sse); free(m->apm); free(m->tstate);
    free(m->tword); free(m->tsparse); free(m->trun);
    free(m->tnibble);  /* v66 */
    free(m->tcc);      /* v67 */
    free(m->tlam);     /* v70: literal-after-match */
    free(m->match_ht); free(m->match_buf);
    free(m->apm2);
}

/* ============================================================================
 * v68: SIMD-утилиты — сравнение строк, смешивание, обновление весов
 * ============================================================================ */
#if KF_USE_SIMD
/* v68: 16-байтовое SIMD-сравнение для match finder */
static inline int kf_match_len_simd(const uint8_t *a, const uint8_t *b,
                                     int cur_len, int max_len) {
    /* Догоняем до 16 байт */
    while (cur_len + 16 <= max_len) {
        __m128i va = _mm_loadu_si128((const __m128i*)(a + cur_len));
        __m128i vb = _mm_loadu_si128((const __m128i*)(b + cur_len));
        __m128i eq = _mm_cmpeq_epi8(va, vb);
        int mask = _mm_movemask_epi8(eq);
        if (mask != 0xFFFF) {
            /* Первый несовпавший байт */
            cur_len += __builtin_ctz(~mask);
            return cur_len < max_len ? cur_len : max_len;
        }
        cur_len += 16;
    }
    /* Хвост побайтово */
    while (cur_len < max_len && a[cur_len] == b[cur_len]) cur_len++;
    return cur_len;
}
#else
static inline int kf_match_len_simd(const uint8_t *a, const uint8_t *b,
                                     int cur_len, int max_len) {
    while (cur_len + 8 <= max_len) {
        uint64_t va8, vb8;
        memcpy(&va8, a + cur_len, 8);
        memcpy(&vb8, b + cur_len, 8);
        if (va8 != vb8) break;
        cur_len += 8;
    }
    while (cur_len < max_len && a[cur_len] == b[cur_len]) cur_len++;
    return cur_len;
}
#endif

#if KF_USE_SIMD
static inline int32_t kf_hsum_epi32(__m128i v) {
    __m128i hi = _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1));
    v = _mm_add_epi32(v, hi);
    hi = _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));
    v = _mm_add_epi32(v, hi);
    return _mm_cvtsi128_si32(v);
}
static inline int32_t kf62_mix(const int16_t *w, const int16_t *s) {
    __m128i w0 = _mm_load_si128((const __m128i*)(w));
    __m128i w1 = _mm_load_si128((const __m128i*)(w + 8));
    __m128i s0 = _mm_load_si128((const __m128i*)(s));
    __m128i s1 = _mm_load_si128((const __m128i*)(s + 8));
    __m128i p0 = _mm_madd_epi16(w0, s0);
    __m128i p1 = _mm_madd_epi16(w1, s1);
    return kf_hsum_epi32(_mm_add_epi32(p0, p1));
}
static inline void kf62_update_weights(int16_t *w, const int16_t *s, int32_t err) {
    __m128i err_v = _mm_set1_epi16((int16_t)(err * 2));
    __m128i min_v = _mm_set1_epi16(-16384);
    __m128i max_v = _mm_set1_epi16(16384);
    /* Блок 0..7 */
    __m128i s0 = _mm_load_si128((const __m128i*)s);
    __m128i d0 = _mm_mulhi_epi16(err_v, s0);  /* (err*2*s) >> 16 ≈ (err*s) >> 15 */
    __m128i w0 = _mm_load_si128((__m128i*)w);
    w0 = _mm_max_epi16(_mm_min_epi16(_mm_add_epi16(w0, d0), max_v), min_v);
    _mm_store_si128((__m128i*)w, w0);
    /* Блок 8..15 */
    __m128i s1 = _mm_load_si128((const __m128i*)(s + 8));
    __m128i d1 = _mm_mulhi_epi16(err_v, s1);
    __m128i w1 = _mm_load_si128((__m128i*)(w + 8));
    w1 = _mm_max_epi16(_mm_min_epi16(_mm_add_epi16(w1, d1), max_v), min_v);
    _mm_store_si128((__m128i*)(w + 8), w1);
}
#else
static inline int32_t kf62_mix(const int16_t *w, const int16_t *s) {
    int32_t sum = 0;
    for (int i = 0; i < KF62_NUM_PREDS; i++)
        sum += (int32_t)w[i] * (int32_t)s[i];
    return sum;
}
static inline void kf62_update_weights(int16_t *w, const int16_t *s, int32_t err) {
    for (int i = 0; i < KF62_NUM_PREDS; i++) {
        int32_t delta = (err * (int32_t)s[i]) >> 15;
        int32_t nw = (int32_t)w[i] + delta;
        if (nw < -16384) nw = -16384;
        if (nw > 16384) nw = 16384;
        w[i] = (int16_t)nw;
    }
}
#endif

/* FNV хеш */
static inline uint32_t kfh(uint32_t h, uint32_t b) {
    return (h ^ b) * 0x01000193u;
}

/* v58: branchless update — убираем условный переход */
static inline void kf_upd(uint16_t *p, int bit, int rate) {
    uint32_t v = *p;
    *p = (uint16_t)(v + ((((uint32_t)bit << 12) - v) >> rate));
}

/* v57: 13 предикторов + APM chain + non-linear quant + per-bit weights.
 * O0-O8, LZ state, word boundary, sparse, run.
 * Нелинейное квантование: stretch-based, 64 бина (больше у 0 и 1).
 * Per-bit weights: отдельные веса для каждой из 8 битовых позиций. */
#define KF51_PROCESS_BYTE(ENCODE)                                           \
do {                                                                        \
    /* Хеши контекстов (Order 1..8) */                                      \
    uint32_t h1 = 0xA1B2C3D4u ^ hist[0];                                   \
    uint32_t h2 = kfh(h1, hist[1]);                                         \
    uint32_t h3 = kfh(h2, hist[2]);                                         \
    uint32_t h4 = kfh(h3, hist[3]);                                         \
    uint32_t h5 = kfh(h4, hist[4]);                                         \
    uint32_t h6 = kfh(h5, hist[5]);                                         \
    uint32_t h7 = kfh(h6, hist[6]);                                         \
    uint32_t h8 = kfh(h7, hist[7]); /* v55: Order-8 */                     \
                                                                            \
    /* Word boundary: пробел/nl/tab → начало нового слова */                \
    int is_wb = (hist[0]==' '||hist[0]=='\n'||hist[0]=='\t'              \
                 ||hist[0]=='\r'||hist[0]==0);                              \
                                                                            \
    /* v72: persistent run counter — отслеживает длинные серии до 63 */      \
    int run_len = run_counter < 63 ? run_counter : 63;                       \
                                                                            \
    uint32_t cx = 1;                                                        \
    for (int b = 7; b >= 0; b--) {                                          \
        /* Индексы в раздельных таблицах */                                  \
        uint32_t i0 = (hist[0] * 256u + cx) & (T0_SIZE - 1);               \
        uint32_t i1 = (h1 * 256u + cx) & (T1_SIZE - 1);                    \
        uint32_t i2 = (h2 * 256u + cx) & (T2_SIZE - 1);                    \
        /* v83: Fibonacci hash (T3=1M, T4/T5/T6=2M, T7/T8=1M) */          \
        uint32_t i3 = ((h3 * 256u + cx) * 0x9E3779B9u) >> 12;              \
        uint32_t i4 = ((h4 * 256u + cx) * 0x9E3779B9u) >> 11;              \
        uint32_t i5 = ((h5 * 256u + cx) * 0x9E3779B9u) >> 11;              \
        uint32_t i6 = ((h6 * 256u + cx) * 0x9E3779B9u) >> 11;              \
        uint32_t i7 = ((h7 * 256u + cx) * 0x9E3779B9u) >> 12;              \
        uint32_t i8 = ((h8 * 256u + cx) * 0x9E3779B9u) >> 12;              \
                                                                            \
        /* State-aware: (lz_state * 65536) + byte * 256 + cx */             \
        uint32_t ist = ((uint32_t)lz_state * 65536u                        \
                        + hist[0] * 256u + cx) & (TSTATE_SIZE - 1);         \
        /* v75: Word model — prev_word_hash на границе слова */             \
        uint32_t iw = is_wb ?                                               \
            ((prev_word_hash * 256u + cx) * 0x9E3779B9u) >> 15              \
            : ((hist[0] * 257u + word_hash) * 256u + cx) & (TWORD_SIZE - 1);\
        /* Sparse skip-gram: hist[0]*33+hist[2] (направленный) */              \
        uint32_t isp = ((hist[0] * 33u + hist[2]) * 256u + cx)             \
                        & (TSPARSE_SIZE - 1);                               \
        /* v55: Run context: run_len * 256 + cx */                          \
        uint32_t irun = ((uint32_t)run_len * 256u + cx)                     \
                         & (TRUN_SIZE - 1);                                 \
                                                                            \
        /* Вероятности (13 предикторов) */                                   \
        uint32_t p0 = mm->t0[i0], p1 = mm->t1[i1];                         \
        uint32_t p2 = mm->t2[i2], p3 = mm->t3[i3];                         \
        uint32_t p4 = mm->t4[i4], p5 = mm->t5[i5];                         \
        uint32_t p6 = mm->t6[i6], p7 = mm->t7[i7];                         \
        uint32_t p8 = mm->t8[i8];                                           \
        uint32_t pst = mm->tstate[ist];                                     \
        uint32_t pw = mm->tword[iw];                                        \
        uint32_t psp = mm->tsparse[isp];                                    \
        uint32_t prun = mm->trun[irun];                                     \
                                                                            \
        /* Логистическое смешивание (13 предикторов) */                    \
        int16_t s[KF_NUM_PREDS];                                            \
        s[0]=kf_stretch(p0);  s[1]=kf_stretch(p1);  s[2]=kf_stretch(p2);   \
        s[3]=kf_stretch(p3);  s[4]=kf_stretch(p4);  s[5]=kf_stretch(p5);   \
        s[6]=kf_stretch(p6);  s[7]=kf_stretch(p7);  s[8]=kf_stretch(p8);   \
        s[9]=kf_stretch(pst); s[10]=kf_stretch(pw);  s[11]=kf_stretch(psp);\
        s[12]=kf_stretch(prun);                                              \
        int32_t logit_mix = 0;                                               \
        /* v70: 16 weight sets: bp(8) × alpha(2) */                       \
        int a_ = (((hist[0]|0x20)>='a')&&((hist[0]|0x20)<='z')) ? 1 : 0;    \
        const int bp_ = (7 - b) * 2 + a_;                                   \
        for (int wi = 0; wi < KF_NUM_PREDS; wi++)                           \
            logit_mix += (int32_t)mm->w[bp_][wi] * (int32_t)s[wi];          \
        logit_mix >>= 8;                                                     \
        uint32_t mx = kf_squash(logit_mix);                                  \
                                                                            \
        /* v73: SSE с 33 бинами — оптимальная конвергенция */                  \
        int q = (int)(mx >> 7);                                             \
        if (q > 32) q = 32;                                                 \
        int sse_cx = ((int)hist[0] >> 4) * 2 + ((int)hist[1] >> 7); /* v75: 32 SSE ctx */ \
        int si = (sse_cx * 8 + (7 - b)) * SSE_Q + q;                        \
        uint32_t sp = mm->sse[si];                                          \
        uint32_t fp = (mx * 3 + sp) >> 2;                                   \
        if (fp < 1) fp = 1; if (fp > 4095) fp = 4095;                      \
        /* v73: APM с 33 бинами */                                          \
        int q2 = (int)(fp >> 7); if (q2 > 32) q2 = 32;                     \
        int is_lz = (lz_state > 0) ? 1 : 0;  /* v59: state-aware */         \
        int api = ((int)cx * 2 + is_lz) * 33 + q2; /* v73: 33 bins */       \
        if (api > (int)(APM_SIZE - 1)) api = (int)(APM_SIZE - 1);           \
        uint32_t ap = mm->apm[api];                                         \
        fp = (fp * 7 + ap) >> 3; /* 87.5% SSE + 12.5% APM */               \
        if (fp < 1) fp = 1; if (fp > 4095) fp = 4095;                      \
                                                                            \
        int bit;                                                             \
        if (ENCODE) {                                                        \
            bit = (byte >> b) & 1;                                           \
            krc_enc_bit(&rc, bit, fp);                                       \
        } else {                                                             \
            bit = krc_dec_bit(&rc, fp);                                      \
            if (bit) byte |= (1u << b);                                      \
        }                                                                    \
                                                                            \
        /* Обучаем контекстные таблицы */                                    \
        kf_upd(&mm->t0[i0], bit, 5);                                        \
        kf_upd(&mm->t1[i1], bit, 5);                                        \
        kf_upd(&mm->t2[i2], bit, 4);                                        \
        kf_upd(&mm->t3[i3], bit, 4);                                        \
        kf_upd(&mm->t4[i4], bit, 3);                                        \
        kf_upd(&mm->t5[i5], bit, 3);                                        \
        kf_upd(&mm->t6[i6], bit, 3);                                        \
        kf_upd(&mm->t7[i7], bit, 3);                                        \
        kf_upd(&mm->t8[i8], bit, 3); /* v55 */                             \
        kf_upd(&mm->tstate[ist], bit, 3);                                   \
        kf_upd(&mm->tword[iw], bit, 4);                                     \
        kf_upd(&mm->tsparse[isp], bit, 4);                                  \
        kf_upd(&mm->trun[irun], bit, 4);                                    \
        kf_upd(&mm->sse[si], bit, 3);  /* v57: rate 3 (стабильная адаптация) */\
        kf_upd(&mm->apm[api], bit, 4);  /* v57: rate 4 (быстрее) */     \
                                                                            \
        /* v57: обучение весов (>>15) */                                    \
        {                                                                    \
            int32_t err = (bit ? 4096 : 0) - (int32_t)mx;                   \
            for (int wi = 0; wi < KF_NUM_PREDS; wi++) {                     \
                int32_t delta = (err * (int32_t)s[wi]) >> 15;               \
                mm->w[bp_][wi] += delta;                 /* v60: per-bit */  \
                if (mm->w[bp_][wi] < -16384) mm->w[bp_][wi] = -16384;       \
                if (mm->w[bp_][wi] > 16384) mm->w[bp_][wi] = 16384;        \
            }                                                                \
        }                                                                    \
                                                                            \
        cx = (cx << 1) | bit;                                                \
    }                                                                        \
    /* v55: 10-state LZ token machine */                                     \
    switch (lz_state) {                                                      \
    case 0: if (byte == 0xFF) lz_state = 1; break;                          \
    case 1:                                                                  \
        if (byte == 0xFF) lz_state = 0;      /* escaped literal */           \
        else if (byte == 0xFE) lz_state = 4; /* short match */              \
        else if (byte == 0xFD) lz_state = 5; /* rep match */                \
        else if (byte == 0xFC) lz_state = 6; /* extended match */           \
        else lz_state = 2;                   /* regular long */              \
        break;                                                               \
    case 2: lz_state = 3; break; /* dist_hi → dist_lo */                    \
    case 3: lz_state = 0; break; /* dist_lo → normal */                     \
    case 4: lz_state = 0; break; /* short dist → normal */                  \
    case 5: lz_state = 0; break; /* rep_byte → normal */                    \
    case 6: lz_state = 7; break; /* ext len → dist[23:16] */               \
    case 7: lz_state = 8; break; /* dist[23:16] → dist[15:8] */            \
    case 8: lz_state = 9; break; /* dist[15:8] → dist[7:0] */              \
    case 9: lz_state = 0; break; /* dist[7:0] → normal */                  \
    default: lz_state = 0; break;                                            \
    }                                                                        \
    /* v72: обновляем persistent run counter */                               \
    run_counter = (byte == hist[0]) ?                                        \
        (run_counter < 255 ? run_counter + 1 : 255) : 0;                     \
    /* v75: сохраняем prev_word_hash + обновляем rolling word hash */   \
    if (byte == ' ' || byte == '\n' || byte == '\t' || byte == '\r'       \
        || byte == 0) {                                                      \
        prev_word_hash = word_hash;                                          \
        word_hash = 0;                                                       \
    } else                                                                   \
        word_hash = word_hash * 37u + byte;                                  \
    /* v71: быстрый сдвиг истории — 64-bit shift вместо memmove */           \
    {                                                                        \
        uint64_t hv64_;                                                      \
        memcpy(&hv64_, hist, 8);                                             \
        hv64_ = (hv64_ << 8) | byte;                                        \
        memcpy(hist, &hv64_, 8);                                             \
    }                                                                        \
} while(0)

/* =====================================================================
 * v66 PROCESS BYTE: 15 предикторов (O0-O8 раздельно), Match, Nibble,
 * SIMD mixing, SSE/APM/APM2 chain
 * ===================================================================== */
#define KF62_PROCESS_BYTE(ENCODE)                                           \
do {                                                                        \
    /* v70: Fibonacci hash macro — лучшее распределение в таблицах */        \
    /* Uses golden ratio constant for optimal hash distribution */           \
    uint32_t h1 = 0xA1B2C3D4u ^ hist[0];                                   \
    uint32_t h2 = kfh(h1, hist[1]);                                         \
    uint32_t h3 = kfh(h2, hist[2]);                                         \
    uint32_t h4 = kfh(h3, hist[3]);                                         \
    uint32_t h5 = kfh(h4, hist[4]);                                         \
    uint32_t h6 = kfh(h5, hist[5]);                                         \
    uint32_t h7 = kfh(h6, hist[6]);                                         \
    uint32_t h8 = kfh(h7, hist[7]);                                         \
    int is_wb = (hist[0]==' '||hist[0]=='\n'||hist[0]=='\t'                  \
                 ||hist[0]=='\r'||hist[0]==0);                               \
    /* v72: persistent run counter — отслеживает длинные серии до 63 */      \
    int run_len = run_counter < 63 ? run_counter : 63;                       \
    uint32_t cx = 1;                                                        \
    /* v82: software prefetch убран — с -O3 hardware prefetcher эффективнее */  \
    for (int b = 7; b >= 0; b--) {                                          \
        uint32_t i0 = (hist[0] * 256u + cx) & (T0_SIZE - 1);               \
        uint32_t i1 = (h1 * 256u + cx) & (T1_SIZE - 1);                    \
        uint32_t i2 = (h2 * 256u + cx) & (T2_SIZE - 1);                    \
        /* v83: Fibonacci hash (T3=1M, T4/T5/T6=2M, T7/T8=1M) */          \
        uint32_t i3 = ((h3 * 256u + cx) * 0x9E3779B9u) >> 12;              \
        uint32_t i4 = ((h4 * 256u + cx) * 0x9E3779B9u) >> 11;              \
        uint32_t i5 = ((h5 * 256u + cx) * 0x9E3779B9u) >> 11;              \
        uint32_t i6 = ((h6 * 256u + cx) * 0x9E3779B9u) >> 11;              \
        uint32_t i7 = ((h7 * 256u + cx) * 0x9E3779B9u) >> 12;              \
        uint32_t i8 = ((h8 * 256u + cx) * 0x9E3779B9u) >> 12;              \
        /* State/word/sparse/run контексты */                                \
        uint32_t ist = ((uint32_t)lz_state * 65536u                        \
                        + hist[0] * 256u + cx) & (TSTATE_SIZE - 1);         \
        /* v75: Word model — prev_word_hash на границе слова */             \
        uint32_t iw = is_wb ?                                               \
            ((prev_word_hash * 256u + cx) * 0x9E3779B9u) >> 15              \
            : ((hist[0] * 257u + word_hash) * 256u + cx) & (TWORD_SIZE - 1);\
        /* Sparse skip-gram: hist[0]*33+hist[2] (направленный) */              \
        uint32_t isp = ((hist[0] * 33u + hist[2]) * 256u + cx)             \
                        & (TSPARSE_SIZE - 1);                               \
        uint32_t irun = ((uint32_t)run_len * 256u + cx)                     \
                         & (TRUN_SIZE - 1);                                 \
        /* v66: Nibble model — верхний полубайт (если уже декодирован) */     \
        uint32_t inib;                                                       \
        if (cx >= 16) {                                                      \
            uint32_t hi_nib = (cx >> 4) & 0xFu;                             \
            inib = (hi_nib * 256u + (uint32_t)hist[0]) * 16u + (cx & 0xFu);\
        } else {                                                             \
            inib = (uint32_t)hist[0] * 256u + cx;                           \
        }                                                                    \
        inib &= (TNIBBLE_SIZE - 1);                                         \
        /* v66: 15 stretch-значений, SIMD-aligned */                        \
        int16_t s_[KF62_PAD] __attribute__((aligned(16)));                  \
        s_[0]  = kf_stretch(mm->t0[i0]);                                    \
        s_[1]  = kf_stretch(mm->t1[i1]);                                    \
        s_[2]  = kf_stretch(mm->t2[i2]);                                    \
        /* v78: PPM-style cold exclusion — untrained O3-O8 contexts             \
         * excluded (stretch=0) to prevent hash collision noise */               \
        { uint32_t pv_ = mm->t3[i3];                                            \
          s_[3] = (pv_ > 1920 && pv_ < 2176) ? 0 : kf_stretch(pv_); }          \
        { uint32_t pv_ = mm->t4[i4];                                            \
          s_[4] = (pv_ > 1920 && pv_ < 2176) ? 0 : kf_stretch(pv_); }          \
        { uint32_t pv_ = mm->t5[i5];                                            \
          s_[5] = (pv_ > 1920 && pv_ < 2176) ? 0 : kf_stretch(pv_); }          \
        { uint32_t pv_ = mm->t6[i6];                                            \
          s_[6] = (pv_ > 1920 && pv_ < 2176) ? 0 : kf_stretch(pv_); }          \
        { uint32_t pv_ = mm->t7[i7];                                            \
          s_[7] = (pv_ > 1920 && pv_ < 2176) ? 0 : kf_stretch(pv_); }          \
        { uint32_t pv_ = mm->t8[i8];                                            \
          s_[8] = (pv_ > 1920 && pv_ < 2176) ? 0 : kf_stretch(pv_); }          \
        s_[9]  = kf_stretch(mm->tstate[ist]);                               \
        s_[10] = kf_stretch(mm->tword[iw]);                                 \
        s_[11] = kf_stretch(mm->tsparse[isp]);                              \
        s_[12] = kf_stretch(mm->trun[irun]);                                \
        /* v71: match model — per-bit предсказание с плавной уверенностью */  \
        if (mm->match_active) {                                              \
            int mbit = (mm->match_byte >> b) & 1;                           \
            int ml = mm->match_len;                                          \
            /* v71: более гранулярная шкала уверенности */                    \
            int conf;                                                        \
            if      (ml < 3)  conf = 120;                                    \
            else if (ml < 5)  conf = 250;                                    \
            else if (ml < 8)  conf = 450;                                    \
            else if (ml < 16) conf = 650;                                    \
            else if (ml < 32) conf = 800;                                    \
            else if (ml < 64) conf = 950;                                    \
            else              conf = 1100;                                   \
            s_[13] = (int16_t)(mbit ? conf : -conf);                        \
        } else if (lam_active) {                                             \
            /* v71: full-byte literal-after-match — match_byte * 256 + cx    \
             * Полный байт вместо nibble для точного предсказания */          \
            uint32_t lam_idx = ((uint32_t)lam_byte * 256u                    \
                               + cx) & (TLAM_SIZE - 1);                      \
            s_[13] = kf_stretch(mm->tlam[lam_idx]);                          \
        } else {                                                             \
            s_[13] = 0;                                                      \
        }                                                                    \
        s_[14] = kf_stretch(mm->tnibble[inib]); /* v66: nibble */            \
        /* v77: Dual-mode character class — text + MTF compatible */          \
        int cc0_ = (hist[0]==0||hist[0]==' '||hist[0]=='\n') ? 0 :            \
                   (hist[0] < 16) ? 1 :                                       \
                   (((hist[0]|0x20)>='a')&&((hist[0]|0x20)<='z')) ? 2 : 3;   \
        int cc1_ = (hist[1]==0||hist[1]==' '||hist[1]=='\n') ? 0 :            \
                   (hist[1] < 16) ? 1 :                                       \
                   (((hist[1]|0x20)>='a')&&((hist[1]|0x20)<='z')) ? 2 : 3;   \
        uint32_t icc = ((uint32_t)(cc0_*4+cc1_)*256u+cx) & (TCC_SIZE-1);     \
        s_[15] = kf_stretch(mm->tcc[icc]); /* v67: char class */              \
        /* v83: PPM-style cascade exclusion — O4/O5 now have 2M tables,        \
         * so their predictions are more reliable, dampen less (>>2 not >>3) */\
        {                                                                    \
            int conf_high = (s_[6] > 50 || s_[6] < -50                       \
                          || s_[7] > 50 || s_[7] < -50                       \
                          || s_[8] > 50 || s_[8] < -50) ? 1 : 0;            \
            if (conf_high) {                                                 \
                s_[3] >>= 3; s_[4] >>= 2; s_[5] >>= 2;                      \
                s_[9] >>= 3;  /* state: LZCM state machine noise */          \
                s_[10] >>= 2; s_[11] >>= 2; s_[12] >>= 2;                  \
                s_[14] >>= 2; s_[15] >>= 2;                                 \
            }                                                                \
        }                                                                    \
        /* v83: 32 weight sets — 8bp × 2alpha × 2rank */                     \
        int a_ = (((hist[0]|0x20)>='a')&&((hist[0]|0x20)<='z')) ? 1 : 0;    \
        int r_ = (hist[0] < 16) ? 1 : 0;                                    \
        const int bp_ = (7 - b) * 4 + a_ * 2 + r_;                          \
        /* v62: SIMD dot-product mixing */                                   \
        int32_t logit_mix = kf62_mix(mm->w[bp_], s_) >> 8;                  \
        uint32_t mx = kf_squash(logit_mix);                                  \
        /* v85: SSE с 9 бинами — finer quantization for better calibration */\
        int q = (int)(mx >> 7); if (q > 32) q = 32;                          \
        /* v77: Finer SSE context — individual 0-15 for MTF ranks */         \
        int h0q_ = hist[0] < 16 ? (int)hist[0]                              \
                                : (16 + ((int)hist[0] >> 5));                \
        /* v78: wider SSE — 96 contexts (was 48) for finer hist[1] resolution */ \
        /* v85: расширенный SSE context — 48 групп (hist[1] >> 6) */     \
        int sse_cx = h0q_ * 2 + ((int)hist[1] >> 7);                        \
        int si = (sse_cx * 8 + (7 - b)) * SSE_Q + q;                        \
        uint32_t sp = mm->sse[si];                                          \
        /* v83: SSE 62.5% — (3mx + 5sp) >> 3 — more SSE weight than         \
         * 50/50 improves probability calibration for BWT data */             \
        uint32_t fp = (mx*3 + sp*5) >> 3;                                    \
        if (fp < 1) fp = 1; if (fp > 4095) fp = 4095;                      \
        /* v80: APM context — is_wb for BWT, is_lz for LZCM */               \
        /* v85: APM 33 бина — fine quantization */                            \
        int q2 = (int)(fp >> 7); if (q2 > 32) q2 = 32;                      \
        int is_lz = (lz_state > 0) ? 1 : 0;                                 \
        int apm_ctx = is_lz | is_wb;                                         \
        int api = ((int)cx * 2 + apm_ctx) * 33 + q2;                         \
        if (api > (int)(APM_SIZE - 1)) api = (int)(APM_SIZE - 1);           \
        uint32_t ap = mm->apm[api];                                         \
        /* v83: APM influence ~69% — (5fp + 11ap) >> 4 — optimal balance     \
         * between mixer+SSE stage and APM's context-aware correction */     \
        fp = (fp*5 + ap*11) >> 4;                                            \
        if (fp < 1) fp = 1; if (fp > 4095) fp = 4095;                      \
        /* v83: APM2 с 5 бинами — coarser, more training per bin */           \
        /* v85: APM2 33 бина */                                               \
        int q3 = (int)(fp >> 7); if (q3 > 32) q3 = 32;                      \
        int a2cx = (((int)(hist[0] >> 4) << 1 | is_wb) * 8 + (7 - b)) & 0x1FF;\
        int a2i = a2cx * 33 + q3;                                            \
        if (a2i > (int)(APM2_SIZE - 1)) a2i = (int)(APM2_SIZE - 1);        \
        uint32_t a2p = mm->apm2[a2i];                                       \
        /* v70: APM2 influence 25% */                                         \
        fp = (fp * 3 + a2p) >> 2;                                           \
        if (fp < 1) fp = 1; if (fp > 4095) fp = 4095;                      \
        int bit;                                                             \
        if ((ENCODE) == 2) {                                                 \
            bit = (byte >> b) & 1;                                           \
        } else if (ENCODE) {                                                 \
            bit = (byte >> b) & 1;                                           \
            krc_enc_bit(&rc, bit, fp);                                       \
        } else {                                                             \
            bit = krc_dec_bit(&rc, fp);                                      \
            if (bit) byte |= (1u << b);                                      \
        }                                                                    \
        /* v80: slower O0 learning (rate 7) for stability — 64K entries,     \
         * 59 avg updates per entry, needs slow adaptation to avoid noise */  \
        kf_upd(&mm->t0[i0], bit, 7);    kf_upd(&mm->t1[i1], bit, 5);      \
        kf_upd(&mm->t2[i2], bit, 5);                                        \
        /* v83: uniform rate 3 for O3-O8 — с увеличенными таблицами           \
         * записи обновляются реже, быстрая адаптация улучшает предсказания */   \
        kf_upd(&mm->t3[i3], bit, 3);                                           \
        kf_upd(&mm->t4[i4], bit, 3);                                           \
        kf_upd(&mm->t5[i5], bit, 3);                                           \
        kf_upd(&mm->t6[i6], bit, 3);                                           \
        kf_upd(&mm->t7[i7], bit, 3);                                           \
        kf_upd(&mm->t8[i8], bit, 3);                                           \
        kf_upd(&mm->tstate[ist], bit, 6);                                   \
        kf_upd(&mm->tword[iw], bit, 6); kf_upd(&mm->tsparse[isp], bit, 6);\
        kf_upd(&mm->trun[irun], bit, 6);                                    \
        kf_upd(&mm->tnibble[inib], bit, 6);  /* v66 */                      \
        kf_upd(&mm->tcc[icc], bit, 6);          /* v67: char class */        \
        /* v71: literal-after-match table update (full-byte index) */          \
        if (lam_active) {                                                    \
            uint32_t lam_i_ = ((uint32_t)lam_byte * 256u                     \
                              + cx) & (TLAM_SIZE - 1);                       \
            kf_upd(&mm->tlam[lam_i_], bit, 4);                              \
        }                                                                    \
        kf_upd(&mm->sse[si], bit, 4);   kf_upd(&mm->apm[api], bit, 4);    \
        kf_upd(&mm->apm2[a2i], bit, 4);                                     \
        /* v62: SIMD weight update */                                        \
        {                                                                    \
            int32_t err_ = (bit ? 4096 : 0) - (int32_t)mx;                 \
            kf62_update_weights(mm->w[bp_], s_, err_);                      \
        }                                                                    \
        cx = (cx << 1) | bit;                                                \
    }                                                                        \
    /* v66: Match Model — 4-байтный хеш для лучших совпадений */             \
    /* v82: match_ht==NULL → skip match model (экономия 16 MB L3 cache) */   \
    if (mm->match_ht) {                                                      \
        size_t mbp = mm->match_buf_pos;                                      \
        mm->match_buf[mbp] = byte;                                           \
        if (mm->match_active) {                                              \
            size_t expected_pos = mm->match_pos + mm->match_len;             \
            if (expected_pos < mbp                                           \
                && mm->match_buf[expected_pos] == byte) {                    \
                mm->match_len++;                                             \
                mm->match_byte = (expected_pos + 1 < mbp + 1)               \
                    ? mm->match_buf[expected_pos + 1] : 0;                  \
            } else {                                                         \
                mm->match_active = 0;                                        \
            }                                                                \
        }                                                                    \
        if (!mm->match_active && mbp >= 4) {                                 \
            /* v71: 2-way set-associative match hash — 4-байтный контекст */ \
            uint32_t mh = ((uint32_t)hist[2] << 24)                         \
                        | ((uint32_t)hist[1] << 16)                          \
                        | ((uint32_t)hist[0] << 8)                           \
                        | (uint32_t)byte;                                    \
            mh = mh * 0x9E3779B1u;                                          \
            mh = (mh ^ (mh >> 12)) & MATCH_HT_MASK;                         \
            uint32_t mh2 = mh * MATCH_HT_WAYS;                              \
            /* Проверяем оба слота (2-way) */                                \
            int match_found = 0;                                             \
            for (int mw = 0; mw < MATCH_HT_WAYS && !match_found; mw++) {    \
                uint32_t mpos = mm->match_ht[mh2 + mw];                     \
                if (mpos > 0 && mpos < (uint32_t)mbp                        \
                    && mm->match_buf[mpos] == byte                           \
                    && mpos >= 1 && mm->match_buf[mpos-1] == hist[0]) {      \
                    mm->match_pos = mpos;                                    \
                    mm->match_len = 1;                                       \
                    mm->match_active = 1;                                    \
                    mm->match_byte = (mpos + 1 < mbp + 1)                   \
                        ? mm->match_buf[mpos + 1] : 0;                      \
                    match_found = 1;                                         \
                }                                                            \
            }                                                                \
            /* LRU-вставка: сдвигаем way[0]→way[1], новый→way[0] */         \
            mm->match_ht[mh2 + 1] = mm->match_ht[mh2];                      \
            mm->match_ht[mh2] = (uint32_t)mbp;                              \
        } else if (mbp >= 4) {                                               \
            uint32_t mh = ((uint32_t)hist[2] << 24)                         \
                        | ((uint32_t)hist[1] << 16)                          \
                        | ((uint32_t)hist[0] << 8)                           \
                        | (uint32_t)byte;                                    \
            mh = mh * 0x9E3779B1u;                                          \
            mh = (mh ^ (mh >> 12)) & MATCH_HT_MASK;                         \
            uint32_t mh2 = mh * MATCH_HT_WAYS;                              \
            mm->match_ht[mh2 + 1] = mm->match_ht[mh2];                      \
            mm->match_ht[mh2] = (uint32_t)mbp;                              \
        }                                                                    \
        mm->match_buf_pos = mbp + 1;                                         \
    }                                                                        \
    /* v79: RLE-aware state machine — 2 состояния для BWT+RLE данных */      \
    if (mm->rle_mode) {                                                      \
        switch (lz_state) {                                                  \
        case 0: if (byte == 0xFF) lz_state = 1; break;                      \
        default: lz_state = 0; break;                                        \
        }                                                                    \
    } else {                                                                 \
    switch (lz_state) {                                                      \
    case 0: if (byte == 0xFF) lz_state = 1; break;                          \
    case 1:                                                                  \
        if (byte == 0xFF) lz_state = 0;                                     \
        else if (byte == 0xFE) lz_state = 4;                                \
        else if (byte == 0xFD) lz_state = 5;                                \
        else if (byte == 0xFC) lz_state = 6;                                \
        else lz_state = 2;                                                   \
        break;                                                               \
    case 2: lz_state = 3; break;                                            \
    case 3: lz_state = 0; break;                                            \
    case 4: lz_state = 0; break;                                            \
    case 5: lz_state = 0; break;                                            \
    case 6: lz_state = 7; break;                                            \
    case 7: lz_state = 8; break;                                            \
    case 8: lz_state = 9; break;                                            \
    case 9: lz_state = 0; break;                                            \
    default: lz_state = 0; break;                                            \
    }                                                                        \
    } /* end rle_mode */                                                     \
    /* v72: обновляем persistent run counter (до сдвига истории) */           \
    run_counter = (byte == hist[0]) ?                                        \
        (run_counter < 255 ? run_counter + 1 : 255) : 0;                     \
    /* v75: сохраняем prev_word_hash + обновляем rolling word hash */   \
    if (byte == ' ' || byte == '\n' || byte == '\t' || byte == '\r'       \
        || byte == 0) {                                                      \
        prev_word_hash = word_hash;                                          \
        word_hash = 0;                                                       \
    } else                                                                   \
        word_hash = word_hash * 37u + byte;                                  \
    /* v71: быстрый сдвиг истории — 64-bit shift вместо memmove */           \
    {                                                                        \
        uint64_t h64_;                                                       \
        memcpy(&h64_, hist, 8);                                              \
        h64_ = (h64_ << 8) | byte;                                          \
        memcpy(hist, &h64_, 8);                                              \
    }                                                                        \
} while(0)
static size_t compress_formula_v51(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_max)
{
    if (input_size == 0 || output_max < 16) return 0;
    KF51M m; if (!kf51_init(&m)) { kf51_destroy(&m); return 0; }
    KF51M *mm = &m;
    KolibriRC rc; krc_enc_init(&rc, output, output_max);
    uint8_t hist[8] = {0};
    int lz_state = 0;
    int run_counter = 0;  /* v72: persistent run counter */
    uint32_t word_hash = 0; /* v74: rolling word hash */
    uint32_t prev_word_hash = 0; /* v75: хеш предыдущего слова */

    for (size_t i = 0; i < input_size; i++) {
        uint8_t byte = input[i];
        KF51_PROCESS_BYTE(1);
    }

    krc_enc_flush(&rc);
    kf51_destroy(&m);
    if (rc.pos >= input_size) return 0;
    return rc.pos;
}

/* --- Декомпрессор v51 --- */
static size_t decompress_formula_v51(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_max,
    size_t original_size)
{
    if (original_size == 0 || original_size > output_max) return 0;
    KF51M m; if (!kf51_init(&m)) { kf51_destroy(&m); return 0; }
    KF51M *mm = &m;
    KolibriRC rc; krc_dec_init(&rc, input, input_size);
    uint8_t hist[8] = {0};
    int lz_state = 0;
    int run_counter = 0;  /* v72: persistent run counter */
    uint32_t word_hash = 0; /* v74: rolling word hash */
    uint32_t prev_word_hash = 0; /* v75: хеш предыдущего слова */

    for (size_t i = 0; i < original_size; i++) {
        uint8_t byte = 0;
        KF51_PROCESS_BYTE(0);
        output[i] = byte;
    }

    kf51_destroy(&m);
    return original_size;
}

/* =====================================================================
 * v66 LZCM: Unified LZ + Context-Mixing encoder
 * Литералы кодируются через CM модель (15 предикторов), матчи через
 * range coder с адаптивными вероятностями.
 * Формат: последовательность range-coded битов:
 *   [match_flag] → if 0: [cm_byte(8 bits)] | if 1: [length][distance]
 * Это архитектура уровня LZMA: LZ parsing + entropy coding.
 * ===================================================================== */

/* v66: кодирование match length (len-4) как 8 бит с адаптивными пробами */
typedef struct {
    uint16_t match_prob[8];     /* v70: P(match) — [lwm*4+char_class] */
    uint16_t len_tree[256];     /* Tree-coded длина для rep (8-bit depth, 255 nodes) */
    uint16_t len_tree_new[256]; /* v68: Tree-coded длина для new match */
    uint16_t dist_slot_tree[32];/* Tree-coded dist slot (5-bit, 31 nodes) */
    uint16_t dist_extra[256];   /* v70: Per-slot context: 16 slots × 16 bit positions */
    /* v66: Rep-match — LZMA-style повторные дистанции */
    uint16_t rep_prob[2];       /* v68: P(rep) — [0] длинный, [1] короткий контекст */
    uint16_t rep0_prob;         /* v68: P(rep0 vs rep1/2/3) */
    uint16_t rep_bits[2];       /* P(bit) для rep1/2/3 индекса */
    int rep[4];                 /* последние 4 дистанции */
    int last_was_match;         /* 0: последний был литерал, 1: матч */
} LZCMState;

static void lzcm_state_init(LZCMState *s) {
    /* v70: 8 контекстов = lwm(2) × char_class(4): letter/digit/space/other */
    for (int i = 0; i < 4; i++) s->match_prob[i] = 1024;      /* после литерала: P(match) ~25% */
    for (int i = 4; i < 8; i++) s->match_prob[i] = 2048;      /* после матча: P(match) ~50% */
    for (int i = 0; i < 256; i++) s->len_tree[i] = 2048;
    for (int i = 0; i < 256; i++) s->len_tree_new[i] = 2048;  /* v68 */
    for (int i = 0; i < 32; i++)  s->dist_slot_tree[i] = 2048;
    for (int i = 0; i < 256; i++) s->dist_extra[i] = 2048;  /* v70: 16 slots × 16 */
    s->rep_prob[0] = 1024;   /* v68: P(rep) */
    s->rep_prob[1] = 1024;
    s->rep0_prob = 3072;    /* v68: P(rep0) ~75% — rep0 самый частый */
    s->rep_bits[0] = 2048;
    s->rep_bits[1] = 2048;
    for (int i = 0; i < 4; i++) s->rep[i] = 0;
    s->last_was_match = 0;
}

/* v68: Match context — 4 контекста: lwm*2 + is_letter */
/* v70: Match context — 8 контекстов: lwm*4 + char_class(letter/digit/space/other) */
#define LZCM_MC(lwm, h0) ((lwm) * 4 + \
    (((h0) | 0x20) >= 'a' && ((h0) | 0x20) <= 'z' ? 0 : \
     ((h0) >= '0' && (h0) <= '9') ? 1 : \
     ((h0) == ' ' || (h0) == '\n' || (h0) == '\t' || (h0) == '\r') ? 2 : 3))

/* Tree-coded длина: MSB→LSB, node = 2*node + bit */
static inline void lzcm_encode_len(KolibriRC *rc, uint16_t *tree, int len_code) {
    int node = 1;
    for (int b = 7; b >= 0; b--) {
        int bit = (len_code >> b) & 1;
        krc_enc_bit(rc, bit, tree[node]);
        kf_upd(&tree[node], bit, 4);
        node = 2 * node + bit;
    }
}

static inline int lzcm_decode_len(KolibriRC *rc, uint16_t *tree) {
    int node = 1;
    for (int b = 7; b >= 0; b--) {
        int bit = krc_dec_bit(rc, tree[node]);
        kf_upd(&tree[node], bit, 4);
        node = 2 * node + bit;
    }
    return node - 256;
}

/* Tree-coded dist slot: 5 бит MSB→LSB */
static inline void lzcm_encode_dist(KolibriRC *rc, LZCMState *s, int dist) {
    int d = dist - 1;
    int nbits = 0;
    if (d > 0) { int tmp = d; while (tmp > 0) { tmp >>= 1; nbits++; } }
    /* Tree-coded nbits (0..20) в 5 битах */
    int node = 1;
    for (int b = 4; b >= 0; b--) {
        int bit = (nbits >> b) & 1;
        krc_enc_bit(rc, bit, s->dist_slot_tree[node]);
        kf_upd(&s->dist_slot_tree[node], bit, 4);
        node = 2 * node + bit;
    }
    /* v70: Extra bits с расширенным per-slot контекстом (16 слотов × 16 бит) */
    int slot_ctx = (nbits > 15) ? 15 : nbits;
    for (int b = nbits - 2; b >= 0; b--) {
        int bit = (d >> b) & 1;
        int bit_pos = nbits - 2 - b;
        if (bit_pos < 16) {
            int ctx = slot_ctx * 16 + bit_pos;
            krc_enc_bit(rc, bit, s->dist_extra[ctx]);
            kf_upd(&s->dist_extra[ctx], bit, 5);
        } else {
            krc_enc_bit(rc, bit, 2048);
        }
    }
}

static inline int lzcm_decode_dist(KolibriRC *rc, LZCMState *s) {
    int node = 1;
    for (int b = 4; b >= 0; b--) {
        int bit = krc_dec_bit(rc, s->dist_slot_tree[node]);
        kf_upd(&s->dist_slot_tree[node], bit, 4);
        node = 2 * node + bit;
    }
    int nbits = node - 32;
    if (nbits == 0) return 1;
    int d = 1;
    /* v70: расширенный контекст дистанций */
    int slot_ctx = (nbits > 15) ? 15 : nbits;
    for (int b = nbits - 2; b >= 0; b--) {
        int bit_pos = nbits - 2 - b;
        int bit;
        if (bit_pos < 16) {
            int ctx = slot_ctx * 16 + bit_pos;
            bit = krc_dec_bit(rc, s->dist_extra[ctx]);
            kf_upd(&s->dist_extra[ctx], bit, 5);
        } else {
            bit = krc_dec_bit(rc, 2048);
        }
        d = (d << 1) | bit;
    }
    return d + 1;
}

/* ============================================================================
 * v66 LZCM — константы и вспомогательные функции
 * ============================================================================ */
#define LZCM_HBITS  22          /* v69: 4M hash entries — меньше коллизий */
#define LZCM_HSIZE  (1u << LZCM_HBITS)
#define LZCM_HMASK  (LZCM_HSIZE - 1u)
#define LZCM_WBITS  22          /* v69: 4MB window — находим далёкие совпадения */
#define LZCM_WSIZE  (1u << LZCM_WBITS)
#define LZCM_WMASK  (LZCM_WSIZE - 1u)
#define LZCM_MIN_MATCH 4
#define LZCM_MAX_MATCH 258
#define LZCM_MAX_CHAIN 2048   /* v74: глубже ищем для лучших матчей */
#define LZCM_NICE_MATCH 258  /* v69: не срезаем рано — ищем максимальные матчи */

/* Поиск лучшего совпадения (rep + хеш-цепочки, без обновления таблиц) */
static inline int lzcm_find_match(
    const uint8_t *input, size_t input_size, size_t pos,
    LZCMState *ls, int32_t *lz_head, int32_t *lz_prev,
    int *out_dist)
{
    int best_len = 0, best_dist = 0;

    /* Rep-match поиск (O(4), очень дёшево) */
    for (int r = 0; r < 4; r++) {
        if (ls->rep[r] > 0 && (size_t)ls->rep[r] <= pos) {
            const uint8_t *a = input + pos;
            const uint8_t *b = input + pos - ls->rep[r];
            int max_len = (int)MIN((size_t)LZCM_MAX_MATCH, input_size - pos);
            int len = 0;
            /* Быстрое 4-байтное сравнение */
            if (max_len >= 4) {
                uint32_t va, vb;
                memcpy(&va, a, 4); memcpy(&vb, b, 4);
                if (va == vb) {
                    len = kf_match_len_simd(a, b, 4, max_len);
                } else {
                    while (len < max_len && a[len] == b[len]) len++;
                }
            } else {
                while (len < max_len && a[len] == b[len]) len++;
            }
            /* Rep: min 3 (дешевле кодировать) */
            if (len >= 3 && len > best_len) {
                best_len = len;
                best_dist = ls->rep[r];
                if (len >= LZCM_NICE_MATCH) { *out_dist = best_dist; return best_len; }
            }
        }
    }

    /* Хеш-цепочки */
    if (pos + 3 < input_size && best_len < LZCM_NICE_MATCH) {
        uint32_t h;
        memcpy(&h, input + pos, 4);
        h = (h * 0x9E3779B1u) >> (32 - LZCM_HBITS);

        /* v71: Адаптивная глубина цепочки — если уже нашли хороший матч,
         * уменьшаем глубину поиска для экономии времени */
        int max_chain = LZCM_MAX_CHAIN;
        if (best_len >= 32) max_chain = 64;
        else if (best_len >= 16) max_chain = 256;
        else if (best_len >= 8) max_chain = 512;

        int32_t cur = lz_head[h];
        int chain = 0;
        while (cur >= 0 && chain < max_chain) {
            size_t candidate = (size_t)cur;
            if (pos > candidate && (pos - candidate) <= LZCM_WSIZE) {
                int len = 0;
                int max_len = (int)MIN((size_t)LZCM_MAX_MATCH, input_size - pos);
                if (max_len >= 4) {
                    uint32_t va, vb;
                    memcpy(&va, input + pos, 4);
                    memcpy(&vb, input + candidate, 4);
                    if (va == vb) {
                        len = kf_match_len_simd(input + pos, input + candidate,
                                                4, max_len);
                    }
                }
                if (len >= LZCM_MIN_MATCH && len > best_len) {
                    best_len = len;
                    best_dist = (int)(pos - candidate);
                    if (len >= LZCM_NICE_MATCH) break;
                }
            }
            cur = lz_prev[candidate & LZCM_WMASK];
            if (cur >= (int32_t)pos) break;
            chain++;
        }
    }

    *out_dist = best_dist;
    return best_len;
}

/* Обновление хеш-цепочки для одной позиции */
static inline void lzcm_update_hash(
    const uint8_t *input, size_t input_size, size_t pos,
    int32_t *lz_head, int32_t *lz_prev)
{
    if (pos + 3 < input_size) {
        uint32_t h;
        memcpy(&h, input + pos, 4);
        h = (h * 0x9E3779B1u) >> (32 - LZCM_HBITS);
        lz_prev[pos & LZCM_WMASK] = lz_head[h];
        lz_head[h] = (int32_t)pos;
    }
}

/* --- v66 LZCM блочный компрессор --- */
static size_t compress_lzcm_v66_block(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_max)
{
    if (input_size == 0 || output_max < 16) return 0;

    /* CM модель */
    KF62M m;
    if (!kf62_init(&m)) { kf62_destroy(&m); return 0; }
    KF62M *mm = &m;
    KolibriRC rc;
    krc_enc_init(&rc, output, output_max);
    uint8_t hist[8] = {0};
    int lz_state = 0;
    /* v70: LZMA-style literal-after-match context */
    uint8_t lam_byte = 0;
    int lam_active = 0;
    int run_counter = 0;  /* v72: persistent run counter */
    uint32_t word_hash = 0; /* v74: rolling word hash */
    uint32_t prev_word_hash = 0; /* v75: хеш предыдущего слова */

    int32_t *lz_head = (int32_t *)malloc(LZCM_HSIZE * sizeof(int32_t));
    int32_t *lz_prev = (int32_t *)malloc(LZCM_WSIZE * sizeof(int32_t));
    if (!lz_head || !lz_prev) {
        free(lz_head); free(lz_prev);
        kf62_destroy(&m); return 0;
    }
    for (size_t j = 0; j < LZCM_HSIZE; j++) lz_head[j] = -1;
    for (size_t j = 0; j < LZCM_WSIZE; j++) lz_prev[j] = -1;

    /* v66: адаптивные вероятности для матчей */
    LZCMState ls;
    lzcm_state_init(&ls);

    size_t i = 0;
    while (i < input_size) {
        /* Поиск лучшего совпадения (rep + хеш-цепочки) */
        int best_len = 0, best_dist = 0;
        best_len = lzcm_find_match(input, input_size, i, &ls,
                                   lz_head, lz_prev, &best_dist);

        /* Обновляем хеш-цепочку для текущей позиции */
        lzcm_update_hash(input, input_size, i, lz_head, lz_prev);

        /* Определяем тип матча (rep или новый) */
        int rep_idx = -1;
        for (int r = 0; r < 4; r++) {
            if (ls.rep[r] == best_dist && ls.rep[r] > 0) { rep_idx = r; break; }
        }

        /* v74: Cost-based rep-match preference — rep0 сохраняет ~3 байта
         * (нет кодирования дистанции), rep1-3 ~1.5 байта */
        if (rep_idx < 0 && best_len >= LZCM_MIN_MATCH) {
            int best_rep_len = 0, best_rep_idx = -1;
            for (int r = 0; r < 4; r++) {
                if (ls.rep[r] > 0 && (size_t)ls.rep[r] <= i) {
                    const uint8_t *a = input + i;
                    const uint8_t *b = input + i - ls.rep[r];
                    int max_len = (int)MIN((size_t)LZCM_MAX_MATCH, input_size - i);
                    int len = 0;
                    while (len < max_len && a[len] == b[len]) len++;
                    if (len >= 3 && len > best_rep_len) {
                        best_rep_len = len;
                        best_rep_idx = r;
                    }
                }
            }
            /* v74: rep0 экономит ~24 бит (match+rep+rep0+len vs match+new+dist+len),
             * rep1-3 экономят ~12 бит.  При CM стоимости ~4 бит/байт,
             * rep0 можно брать даже на 3 байта короче */
            int rep_advantage = (best_rep_idx == 0) ? 3 : 1;
            if (best_rep_idx >= 0 && best_rep_len >= best_len - rep_advantage) {
                best_len = best_rep_len;
                best_dist = ls.rep[best_rep_idx];
                rep_idx = best_rep_idx;
            }
        }

        /* v74: Дистанция-зависимый минимум длины для новых матчей.
         * Ослаблены пороги: при CM стоимости ~4 бит/байт, матч выгоден
         * если len*4 > dist_coding_cost. Для dist=512 это len >= 5,
         * для dist=4096 — len >= 6, для dist=32K — len >= 7. */
        int min_match;
        if (rep_idx >= 0) {
            min_match = 3;
        } else {
            min_match = LZCM_MIN_MATCH;  /* 4 */
            if (best_dist > 256)   min_match = 5;
            if (best_dist > 4096)  min_match = 6;
            if (best_dist > 32768) min_match = 7;
            if (best_dist > 131072) min_match = 8;
        }

        if (best_len >= min_match) {
            /* v66: Lazy parsing — проверяем, есть ли лучший матч на i+1 */
            if (best_len < LZCM_NICE_MATCH && i + 1 < input_size) {
                int lazy_len = 0, lazy_dist = 0;
                lazy_len = lzcm_find_match(input, input_size, i + 1, &ls,
                                           lz_head, lz_prev, &lazy_dist);
                int lazy_rep = -1;
                for (int r = 0; r < 4; r++) {
                    if (ls.rep[r] == lazy_dist && ls.rep[r] > 0)
                        { lazy_rep = r; break; }
                }
                int lazy_min = (lazy_rep >= 0) ? 3 : LZCM_MIN_MATCH;

                /* v74: Rep-aware lazy scoring — rep-match на lazy позиции дешевле,
                 * поэтому порог снижается. Также учитываем стоимость
                 * кодирования дистанции: rep экономит ~12-24 бит */
                int lazy_bonus = 0;
                if (lazy_rep >= 0 && rep_idx < 0) lazy_bonus = 2; /* lazy=rep, orig=new */
                else if (lazy_rep >= 0 && rep_idx >= 0) lazy_bonus = 0; /* both rep */
                else if (lazy_rep < 0 && rep_idx >= 0) lazy_bonus = -2; /* lazy=new, orig=rep */
                int lazy_thresh = (best_len < 8) ? (best_len - lazy_bonus) : (best_len + 1 - lazy_bonus);
                if (lazy_len >= lazy_min && lazy_len > lazy_thresh) {
                    /* Литерал на позиции i */
                    krc_enc_bit(&rc, 0, ls.match_prob[LZCM_MC(ls.last_was_match, hist[0])]);
                    kf_upd(&ls.match_prob[LZCM_MC(ls.last_was_match, hist[0])], 0, 4);
                    /* v70: literal-after-match context */
                    lam_active = (ls.last_was_match && ls.rep[0] > 0 && i >= (size_t)ls.rep[0]);
                    lam_byte = lam_active ? input[i - ls.rep[0]] : 0;
                    ls.last_was_match = 0;
                    uint8_t byte = input[i];
                    KF62_PROCESS_BYTE(1);
                    lam_active = 0;
                    lz_state = 0;
                    i++;

                    /* Обновляем хеш для новой позиции */
                    lzcm_update_hash(input, input_size, i, lz_head, lz_prev);

                    best_len = lazy_len;
                    best_dist = lazy_dist;
                    rep_idx = lazy_rep;
                    min_match = lazy_min;

                    /* v66: Double lazy — проверяем i+2 тоже */
                    if (best_len < LZCM_NICE_MATCH && i + 1 < input_size) {
                        int lazy2_len = 0, lazy2_dist = 0;
                        lazy2_len = lzcm_find_match(input, input_size, i + 1,
                                                    &ls, lz_head, lz_prev,
                                                    &lazy2_dist);
                        int lazy2_rep = -1;
                        for (int r = 0; r < 4; r++) {
                            if (ls.rep[r] == lazy2_dist && ls.rep[r] > 0)
                                { lazy2_rep = r; break; }
                        }
                        int lazy2_min = (lazy2_rep >= 0) ? 3 : LZCM_MIN_MATCH;
                        /* v74: rep-aware lazy2 scoring */
                        int lazy2_bonus = 0;
                        if (lazy2_rep >= 0 && rep_idx < 0) lazy2_bonus = 2;
                        else if (lazy2_rep < 0 && rep_idx >= 0) lazy2_bonus = -2;
                        int lazy2_thresh = (best_len < 8) ? (best_len - lazy2_bonus) : (best_len + 1 - lazy2_bonus);
                        if (lazy2_len >= lazy2_min && lazy2_len > lazy2_thresh) {
                            /* Ещё один литерал */
                            krc_enc_bit(&rc, 0, ls.match_prob[LZCM_MC(ls.last_was_match, hist[0])]);
                            kf_upd(&ls.match_prob[LZCM_MC(ls.last_was_match, hist[0])], 0, 4);
                            lam_active = (ls.last_was_match && ls.rep[0] > 0 && i >= (size_t)ls.rep[0]);
                            lam_byte = lam_active ? input[i - ls.rep[0]] : 0;
                            ls.last_was_match = 0;
                            byte = input[i];
                            KF62_PROCESS_BYTE(1);
                            lam_active = 0;
                            lz_state = 0;
                            i++;
                            lzcm_update_hash(input, input_size, i,
                                             lz_head, lz_prev);
                            best_len = lazy2_len;
                            best_dist = lazy2_dist;
                            rep_idx = lazy2_rep;
                            min_match = lazy2_min;

                            /* v70: Triple lazy — проверяем i+3 для коротких матчей */
                            if (best_len < 16 && best_len < LZCM_NICE_MATCH && i + 1 < input_size) {
                                int lazy3_len = 0, lazy3_dist = 0;
                                lazy3_len = lzcm_find_match(input, input_size, i + 1,
                                                            &ls, lz_head, lz_prev,
                                                            &lazy3_dist);
                                int lazy3_rep = -1;
                                for (int r = 0; r < 4; r++) {
                                    if (ls.rep[r] == lazy3_dist && ls.rep[r] > 0)
                                        { lazy3_rep = r; break; }
                                }
                                int lazy3_min = (lazy3_rep >= 0) ? 3 : LZCM_MIN_MATCH;
                                /* v74: rep-aware lazy3 scoring */
                                int lazy3_bonus = 0;
                                if (lazy3_rep >= 0 && rep_idx < 0) lazy3_bonus = 2;
                                else if (lazy3_rep < 0 && rep_idx >= 0) lazy3_bonus = -2;
                                int lazy3_thresh = (best_len < 8) ? (best_len - lazy3_bonus) : (best_len + 1 - lazy3_bonus);
                                if (lazy3_len >= lazy3_min && lazy3_len > lazy3_thresh) {
                                    krc_enc_bit(&rc, 0, ls.match_prob[LZCM_MC(ls.last_was_match, hist[0])]);
                                    kf_upd(&ls.match_prob[LZCM_MC(ls.last_was_match, hist[0])], 0, 4);
                                    lam_active = (ls.last_was_match && ls.rep[0] > 0 && i >= (size_t)ls.rep[0]);
                                    lam_byte = lam_active ? input[i - ls.rep[0]] : 0;
                                    ls.last_was_match = 0;
                                    byte = input[i];
                                    KF62_PROCESS_BYTE(1);
                                    lam_active = 0;
                                    lz_state = 0;
                                    i++;
                                    lzcm_update_hash(input, input_size, i, lz_head, lz_prev);
                                    best_len = lazy3_len;
                                    best_dist = lazy3_dist;
                                    rep_idx = lazy3_rep;
                                    min_match = lazy3_min;
                                }
                            }
                        }
                    }
                }
            }

            /* Кодируем матч: match_bit → rep_bit → (rep: idx+len_from3 / new: len_from4+dist) */
            krc_enc_bit(&rc, 1, ls.match_prob[LZCM_MC(ls.last_was_match, hist[0])]);
            kf_upd(&ls.match_prob[LZCM_MC(ls.last_was_match, hist[0])], 1, 4);

            if (rep_idx >= 0) {
                /* v68: Rep-match: rep_bit=1 + is_rep0 + (если нет: 2 бита для rep1/2/3) + длина */
                krc_enc_bit(&rc, 1, ls.rep_prob[0]);
                kf_upd(&ls.rep_prob[0], 1, 4);
                if (rep_idx == 0) {
                    krc_enc_bit(&rc, 1, ls.rep0_prob);
                    kf_upd(&ls.rep0_prob, 1, 4);
                } else {
                    krc_enc_bit(&rc, 0, ls.rep0_prob);
                    kf_upd(&ls.rep0_prob, 0, 4);
                    int ri = rep_idx - 1;  /* 0,1,2 → rep1,rep2,rep3 */
                    krc_enc_bit(&rc, (ri >> 1) & 1, ls.rep_bits[0]);
                    kf_upd(&ls.rep_bits[0], (ri >> 1) & 1, 4);
                    krc_enc_bit(&rc, ri & 1, ls.rep_bits[1]);
                    kf_upd(&ls.rep_bits[1], ri & 1, 4);
                }
                lzcm_encode_len(&rc, ls.len_tree, best_len - 3);
            } else {
                /* Новая дистанция: rep_bit=0 + длина от 4 + дистанция */
                krc_enc_bit(&rc, 0, ls.rep_prob[0]);
                kf_upd(&ls.rep_prob[0], 0, 4);
                lzcm_encode_len(&rc, ls.len_tree_new, best_len - LZCM_MIN_MATCH);
                lzcm_encode_dist(&rc, &ls, best_dist);
                /* Обновляем rep-массив (сдвиг) */
                ls.rep[3] = ls.rep[2];
                ls.rep[2] = ls.rep[1];
                ls.rep[1] = ls.rep[0];
                ls.rep[0] = best_dist;
            }

            /* v67: CM обновление во время матчей */
            for (int j = 0; j < best_len; j++) {
                /* Хеш-цепочка для будущих матчей (все позиции, как в deflate) */
                if (j > 0 && (i + j + 3) < input_size) {
                    lzcm_update_hash(input, input_size, i + (size_t)j,
                                     lz_head, lz_prev);
                }
                /* v77: Матчи <=32 обучают CM, >32 — только hist */
                if (best_len <= 32) {
                    uint8_t byte = input[i + j];
                    KF62_PROCESS_BYTE(2);
                    lz_state = 0;
                } else {
                    mm->match_buf[mm->match_buf_pos++] = input[i + j];
                    /* v72: обновляем run_counter и в match-copy пути */
                    run_counter = (input[i + j] == hist[0]) ?
                        (run_counter < 255 ? run_counter + 1 : 255) : 0;
                    /* v75: prev_word_hash + word_hash в match-copy */
                    if (input[i+j]==' '||input[i+j]=='\n'||input[i+j]=='\t'
                        ||input[i+j]=='\r'||input[i+j]==0) {
                        prev_word_hash = word_hash;
                        word_hash = 0;
                    } else
                        word_hash = word_hash * 37u + input[i + j];
                    /* v71: быстрый сдвиг истории */
                    {
                        uint64_t h64m_;
                        memcpy(&h64m_, hist, 8);
                        h64m_ = (h64m_ << 8) | input[i + j];
                        memcpy(hist, &h64m_, 8);
                    }
                }
            }
            if (best_len > 32) mm->match_active = 0;  /* v77 */
            ls.last_was_match = 1;
            i += best_len;
        } else {
            /* Кодируем литерал через CM модель */
            krc_enc_bit(&rc, 0, ls.match_prob[LZCM_MC(ls.last_was_match, hist[0])]);
            kf_upd(&ls.match_prob[LZCM_MC(ls.last_was_match, hist[0])], 0, 4);

            /* v70: literal-after-match context */
            lam_active = (ls.last_was_match && ls.rep[0] > 0 && i >= (size_t)ls.rep[0]);
            lam_byte = lam_active ? input[i - ls.rep[0]] : 0;

            uint8_t byte = input[i];
            KF62_PROCESS_BYTE(1);
            lam_active = 0;
            lz_state = 0;
            ls.last_was_match = 0;
            i++;
        }
    }

    krc_enc_flush(&rc);
    kf62_destroy(&m);
    free(lz_head);
    free(lz_prev);

    if (rc.pos >= input_size) return 0;
    return rc.pos;
}

/* --- v66 LZCM блочный декомпрессор --- */
static size_t decompress_lzcm_v66_block(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_max,
    size_t original_size)
{
    if (original_size == 0 || original_size > output_max) return 0;

    KF62M m;
    if (!kf62_init(&m)) { kf62_destroy(&m); return 0; }
    KF62M *mm = &m;
    KolibriRC rc;
    krc_dec_init(&rc, input, input_size);
    uint8_t hist[8] = {0};
    int lz_state = 0;
    /* v70: LZMA-style literal-after-match context */
    uint8_t lam_byte = 0;
    int lam_active = 0;
    int run_counter = 0;  /* v72: persistent run counter */
    uint32_t word_hash = 0; /* v74: rolling word hash */
    uint32_t prev_word_hash = 0; /* v75: хеш предыдущего слова */

    LZCMState ls;
    lzcm_state_init(&ls);

    size_t i = 0;
    while (i < original_size) {
        int is_match = krc_dec_bit(&rc, ls.match_prob[LZCM_MC(ls.last_was_match, hist[0])]);
        kf_upd(&ls.match_prob[LZCM_MC(ls.last_was_match, hist[0])], is_match, 4);

        if (is_match) {
            /* v66: Новый формат — rep_bit перед длиной */
            int is_rep = krc_dec_bit(&rc, ls.rep_prob[0]);
            kf_upd(&ls.rep_prob[0], is_rep, 4);

            int dist, len;
            if (is_rep) {
                /* v68: Rep-match: is_rep0 + (если нет: 2 бита) + длина от 3 */
                int is_rep0 = krc_dec_bit(&rc, ls.rep0_prob);
                kf_upd(&ls.rep0_prob, is_rep0, 4);
                int ri;
                if (is_rep0) {
                    ri = 0;
                } else {
                    int bit0 = krc_dec_bit(&rc, ls.rep_bits[0]);
                    kf_upd(&ls.rep_bits[0], bit0, 4);
                    int bit1 = krc_dec_bit(&rc, ls.rep_bits[1]);
                    kf_upd(&ls.rep_bits[1], bit1, 4);
                    ri = (bit0 << 1 | bit1) + 1;
                }
                dist = ls.rep[ri];
                len = lzcm_decode_len(&rc, ls.len_tree) + 3;
            } else {
                /* Новый матч: длина от 4 + дистанция */
                len = lzcm_decode_len(&rc, ls.len_tree_new) + LZCM_MIN_MATCH;
                dist = lzcm_decode_dist(&rc, &ls);
                ls.rep[3] = ls.rep[2];
                ls.rep[2] = ls.rep[1];
                ls.rep[1] = ls.rep[0];
                ls.rep[0] = dist;
            }

            if (dist <= 0 || (size_t)dist > i || i + (size_t)len > original_size) {
                kf62_destroy(&m);
                return 0;
            }
            for (int j = 0; j < len; j++) {
                output[i + j] = output[i + j - dist];
                /* v77: Матчи <=32 обучают CM, >32 — только hist */
                if (len <= 32) {
                    uint8_t byte = output[i + j];
                    KF62_PROCESS_BYTE(2);
                    lz_state = 0;
                } else {
                    mm->match_buf[mm->match_buf_pos++] = output[i + j];
                    /* v72: обновляем run_counter и в match-copy пути */
                    run_counter = (output[i + j] == hist[0]) ?
                        (run_counter < 255 ? run_counter + 1 : 255) : 0;
                    /* v75: prev_word_hash + word_hash в match-copy */
                    if (output[i+j]==' '||output[i+j]=='\n'||output[i+j]=='\t'
                        ||output[i+j]=='\r'||output[i+j]==0) {
                        prev_word_hash = word_hash;
                        word_hash = 0;
                    } else
                        word_hash = word_hash * 37u + output[i + j];
                    /* v71: быстрый сдвиг истории */
                    {
                        uint64_t h64d_;
                        memcpy(&h64d_, hist, 8);
                        h64d_ = (h64d_ << 8) | output[i + j];
                        memcpy(hist, &h64d_, 8);
                    }
                }
            }
            if (len > 32) mm->match_active = 0;  /* v77 */
            ls.last_was_match = 1;
            i += (size_t)len;
        } else {
            /* v70: literal-after-match context for decompressor */
            lam_active = (ls.last_was_match && ls.rep[0] > 0 && i >= (size_t)ls.rep[0]);
            lam_byte = lam_active ? output[i - ls.rep[0]] : 0;

            uint8_t byte = 0;
            KF62_PROCESS_BYTE(0);
            lam_active = 0;
            output[i] = byte;
            lz_state = 0;
            ls.last_was_match = 0;
            i++;
        }
    }

    kf62_destroy(&m);
    return original_size;
}

/* --- v66 LZCM блочная обёртка (аналог compress_formula_v62) --- */
static size_t compress_lzcm_v66(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_max)
{
    if (input_size == 0 || output_max < 16) return 0;

    size_t nblocks = (input_size + KF62_BLOCK_SIZE - 1) / KF62_BLOCK_SIZE;
    if (nblocks > 65535) nblocks = 65535;

    /* v66: Compact single-block format — экономим 5 байт на заголовке */
    if (nblocks == 1) {
        /* Однобрачный формат: 1 байт флаг (0x00) + сжатые данные */
        uint8_t *bbuf = (uint8_t *)malloc(input_size + 256);
        if (!bbuf) return 0;
        size_t csize = compress_lzcm_v66_block(input, input_size,
                                                 bbuf, input_size + 256);
        if (csize == 0) {
            /* Несжимаемый блок: флаг + raw данные */
            if (1 + input_size >= output_max || 1 + input_size >= input_size) {
                free(bbuf); return 0;
            }
            output[0] = 0x01; /* raw single block */
            memcpy(output + 1, input, input_size);
            free(bbuf);
            return 1 + input_size;
        }
        if (1 + csize >= output_max || 1 + csize >= input_size) {
            free(bbuf); return 0;
        }
        output[0] = 0x00; /* compressed single block */
        memcpy(output + 1, bbuf, csize);
        free(bbuf);
        return 1 + csize;
    }

    /* Многоблочный формат: флаг 0x02 + count + sizes + data */
    size_t hdr_size = 1 + 2 + nblocks * 4;
    if (hdr_size >= output_max) return 0;

    uint8_t **bbufs = (uint8_t **)calloc(nblocks, sizeof(uint8_t*));
    size_t *bsizes = (size_t *)calloc(nblocks, sizeof(size_t));
    size_t *bcsizes = (size_t *)calloc(nblocks, sizeof(size_t));
    if (!bbufs || !bsizes || !bcsizes) goto lzcm_fail;

    for (size_t bi = 0; bi < nblocks; bi++) {
        size_t off = bi * KF62_BLOCK_SIZE;
        bsizes[bi] = MIN((size_t)KF62_BLOCK_SIZE, input_size - off);
        bbufs[bi] = (uint8_t *)malloc(bsizes[bi] + 256);
        if (!bbufs[bi]) goto lzcm_fail;
        bcsizes[bi] = compress_lzcm_v66_block(
            input + off, bsizes[bi], bbufs[bi], bsizes[bi] + 256);
    }

    /* Заголовок блоков (многоблочный) */
    output[0] = 0x02; /* multi-block flag */
    output[1] = (uint8_t)(nblocks & 0xFF);
    output[2] = (uint8_t)((nblocks >> 8) & 0xFF);
    size_t pos = 3;
    for (size_t bi = 0; bi < nblocks; bi++) {
        uint32_t cs = (uint32_t)bcsizes[bi];
        memcpy(output + pos, &cs, 4);
        pos += 4;
    }

    for (size_t bi = 0; bi < nblocks; bi++) {
        if (bcsizes[bi] == 0) {
            if (pos + bsizes[bi] > output_max) goto lzcm_fail;
            memcpy(output + pos, input + bi * KF62_BLOCK_SIZE, bsizes[bi]);
            pos += bsizes[bi];
        } else {
            if (pos + bcsizes[bi] > output_max) goto lzcm_fail;
            memcpy(output + pos, bbufs[bi], bcsizes[bi]);
            pos += bcsizes[bi];
        }
    }

    for (size_t bi = 0; bi < nblocks; bi++) free(bbufs[bi]);
    free(bbufs); free(bsizes); free(bcsizes);

    if (pos >= input_size) return 0;
    return pos;

lzcm_fail:
    if (bbufs) { for (size_t bi = 0; bi < nblocks; bi++) free(bbufs[bi]); }
    free(bbufs); free(bsizes); free(bcsizes);
    return 0;
}

/* --- v66 LZCM декомпрессор (блочная обёртка) --- */
static size_t decompress_lzcm_v66(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_max,
    size_t original_size)
{
    if (original_size == 0 || input_size < 1) return 0;

    uint8_t flag = input[0];

    /* v66: Compact single-block format */
    if (flag == 0x00) {
        /* Compressed single block */
        return decompress_lzcm_v66_block(
            input + 1, input_size - 1, output, output_max, original_size);
    }
    if (flag == 0x01) {
        /* Raw single block */
        if (input_size - 1 < original_size || original_size > output_max) return 0;
        memcpy(output, input + 1, original_size);
        return original_size;
    }

    /* Multi-block format (flag == 0x02) */
    if (input_size < 3) return 0;
    uint16_t nblocks = (uint16_t)input[1] | ((uint16_t)input[2] << 8);
    if (nblocks == 0 || input_size < 3 + (size_t)nblocks * 4) return 0;

    size_t pos = 3;
    uint32_t *bcsizes = (uint32_t *)malloc(nblocks * sizeof(uint32_t));
    if (!bcsizes) return 0;

    for (int bi = 0; bi < nblocks; bi++) {
        memcpy(&bcsizes[bi], input + pos, 4);
        pos += 4;
    }

    size_t out_pos = 0;
    for (int bi = 0; bi < nblocks; bi++) {
        size_t bsize = MIN((size_t)KF62_BLOCK_SIZE, original_size - out_pos);
        if (bcsizes[bi] == 0) {
            if (pos + bsize > input_size || out_pos + bsize > output_max) {
                free(bcsizes); return 0;
            }
            memcpy(output + out_pos, input + pos, bsize);
            pos += bsize;
        } else {
            size_t dec = decompress_lzcm_v66_block(
                input + pos, bcsizes[bi],
                output + out_pos, output_max - out_pos, bsize);
            if (dec != bsize) { free(bcsizes); return 0; }
            pos += bcsizes[bi];
        }
        out_pos += bsize;
    }

    free(bcsizes);
    return out_pos;
}

/* =====================================================================
 * v62 COMPRESS/DECOMPRESS: merged model + SIMD + multi-threading
 * =====================================================================
 * Формат: [num_blocks:2] [block_csizes:4*N] [block_data...]
 * Каждый блок ≤64KB, независимая модель, параллельное сжатие.
 * ===================================================================== */

/* --- Блочный компрессор v62 (один блок, одна модель) --- */
__attribute__((hot))
static size_t compress_formula_v62_block_ex(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_max, int rle)
{
    if (input_size == 0 || output_max < 16) return 0;
    KF62M m; if (!kf62_init(&m)) { kf62_destroy(&m); return 0; }
    m.rle_mode = rle ? 1 : 0;  /* v79: RLE-aware state machine */
    KF62M *mm = &m;
    KolibriRC rc; krc_enc_init(&rc, output, output_max);
    uint8_t hist[8] = {0};
    int lz_state = 0;
    uint8_t lam_byte = 0; int lam_active = 0; /* v70: unused in pure CM */
    int run_counter = 0;  /* v72: persistent run counter */
    uint32_t word_hash = 0; /* v74: rolling word hash */
    uint32_t prev_word_hash = 0; /* v75: хеш предыдущего слова */

    for (size_t i = 0; i < input_size; i++) {
        uint8_t byte = input[i];
        KF62_PROCESS_BYTE(1);
    }

    krc_enc_flush(&rc);
    kf62_destroy(&m);
    if (rc.pos >= input_size) return 0;
    return rc.pos;
}

static size_t compress_formula_v62_block(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_max)
{
    return compress_formula_v62_block_ex(input, input_size, output, output_max, 0);
}

/* v82: Fast BWT block — без match hash table (экономия 16 MB L3 cache).
 * Для BWT+MTF+RLE данных match model практически бесполезен:
 * BWT перемешивает порядок байтов, уничтожая последовательные паттерны. */
__attribute__((hot))
static size_t compress_formula_v62_block_fast(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_max)
{
    if (input_size == 0 || output_max < 16) return 0;
    KF62M m; if (!kf62_init_fast(&m)) { kf62_destroy(&m); return 0; }
    KF62M *mm = &m;
    KolibriRC rc; krc_enc_init(&rc, output, output_max);
    uint8_t hist[8] = {0};
    int lz_state = 0;
    uint8_t lam_byte = 0; int lam_active = 0;
    int run_counter = 0;
    uint32_t word_hash = 0;
    uint32_t prev_word_hash = 0;

    for (size_t i = 0; i < input_size; i++) {
        uint8_t byte = input[i];
        KF62_PROCESS_BYTE(1);
    }

    krc_enc_flush(&rc);
    kf62_destroy(&m);
    if (rc.pos >= input_size) return 0;
    return rc.pos;
}

/* v82: Fast BWT decompressor — symmetric with fast compressor */
static size_t decompress_formula_v62_block_fast(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_max,
    size_t original_size)
{
    if (original_size == 0 || original_size > output_max) return 0;
    KF62M m; if (!kf62_init_fast(&m)) { kf62_destroy(&m); return 0; }
    KF62M *mm = &m;
    KolibriRC rc; krc_dec_init(&rc, input, input_size);
    uint8_t hist[8] = {0};
    int lz_state = 0;
    uint8_t lam_byte = 0; int lam_active = 0;
    int run_counter = 0;
    uint32_t word_hash = 0;
    uint32_t prev_word_hash = 0;

    for (size_t i = 0; i < original_size; i++) {
        uint8_t byte = 0;
        KF62_PROCESS_BYTE(0);
        output[i] = byte;
    }

    kf62_destroy(&m);
    return original_size;
}

/* --- Блочный декомпрессор v62 --- */
static size_t decompress_formula_v62_block_ex(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_max,
    size_t original_size, int rle)
{
    if (original_size == 0 || original_size > output_max) return 0;
    KF62M m; if (!kf62_init(&m)) { kf62_destroy(&m); return 0; }
    m.rle_mode = rle ? 1 : 0;  /* v79: RLE-aware state machine */
    KF62M *mm = &m;
    KolibriRC rc; krc_dec_init(&rc, input, input_size);
    uint8_t hist[8] = {0};
    int lz_state = 0;
    uint8_t lam_byte = 0; int lam_active = 0; /* v70: unused in pure CM */
    int run_counter = 0;  /* v72: persistent run counter */
    uint32_t word_hash = 0; /* v74: rolling word hash */
    uint32_t prev_word_hash = 0; /* v75: хеш предыдущего слова */

    for (size_t i = 0; i < original_size; i++) {
        uint8_t byte = 0;
        KF62_PROCESS_BYTE(0);
        output[i] = byte;
    }

    kf62_destroy(&m);
    return original_size;
}

static size_t decompress_formula_v62_block(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_max,
    size_t original_size)
{
    return decompress_formula_v62_block_ex(input, input_size, output, output_max,
                                            original_size, 0);
}

/* ============================================================================
 * v85: TURBO bit-level order-1+2 adaptive binary tree coder
 *
 * Замена CM для BWT+MTF+RLE данных с ~2.6× ускорением:
 *   CM:    8 bit-ops × 16 predictors × SSE/APM/APM2 = ~180ms на 500KB
 *   TURBO: 8 bit-ops × 2 trees (o1+o2 blend)       = ~69ms  на 500KB
 *
 * Модель: order-1 (128KB) + sparse order-2 (1KB) = 129KB → L2 cache.
 *   o1[256][256]: полный order-1 контекст (предыдущий байт)
 *   o2[2][256]:   sparse order-2 (MSB пред-предыдущего байта)
 * Финальная вероятность: (p1 + p2) >> 1.
 * Использует ПРОВЕРЕННЫЙ krc_enc_bit/krc_dec_bit.
 * ============================================================================ */

#define TURBO_RATE  5   /* скорость адаптации: prob += delta >> 5 (1/32) */

/* Адаптивная модель order-1+2 (sparse) */
typedef struct {
    uint16_t o1[256][256];   /* 128KB: order-1 деревья */
    uint16_t o2[2][256];     /* 1KB: sparse order-2 (MSB prev2) */
} TurboModel;

static TurboModel *turbo_model_create(void) {
    TurboModel *m = (TurboModel *)malloc(sizeof(TurboModel));
    if (!m) return NULL;
    /* Инициализация всех вероятностей = 2048 (50%) */
    uint16_t *p = (uint16_t *)m;
    size_t count = sizeof(TurboModel) / sizeof(uint16_t);
    for (size_t i = 0; i < count; i++) p[i] = 2048;
    return m;
}

/* --- Turbo compress: bit-level order-1+2 (sparse) binary tree --- */
__attribute__((hot))
static size_t compress_turbo_block(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_max)
{
    if (input_size == 0 || output_max < 16) return 0;
    TurboModel *m = turbo_model_create();
    if (!m) return 0;

    KolibriRC rc;
    krc_enc_init(&rc, output, output_max);

    int ctx = 0, ctx2 = 0;
    for (size_t i = 0; i < input_size; i++) {
        int sym = input[i];
        uint16_t *__restrict__ tree = m->o1[ctx];
        uint16_t *__restrict__ tree2 = m->o2[ctx2];
        int node = 1;
        for (int b = 7; b >= 0; b--) {
            int bit = (sym >> b) & 1;
            uint32_t p1 = tree[node];
            uint32_t p2 = tree2[node];
            uint32_t p = (p1 + p2) >> 1;
            if (p < 1) p = 1; if (p > 4095) p = 4095;
            krc_enc_bit(&rc, bit, p);
            if (bit) {
                tree[node] += (uint16_t)((4096 - p1) >> TURBO_RATE);
                tree2[node] += (uint16_t)((4096 - p2) >> TURBO_RATE);
            } else {
                tree[node] -= (uint16_t)(p1 >> TURBO_RATE);
                tree2[node] -= (uint16_t)(p2 >> TURBO_RATE);
            }
            node = node * 2 + bit;
        }
        ctx2 = ctx >> 7;
        ctx = sym;
    }

    krc_enc_flush(&rc);
    free(m);
    if (rc.pos >= input_size) return 0;
    return rc.pos;
}

/* --- Turbo decompress: bit-level order-1+2 (sparse) binary tree --- */
__attribute__((hot))
static size_t decompress_turbo_block(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_max,
    size_t original_size)
{
    if (original_size == 0 || original_size > output_max) return 0;
    TurboModel *m = turbo_model_create();
    if (!m) return 0;

    KolibriRC rc;
    krc_dec_init(&rc, input, input_size);

    int ctx = 0, ctx2 = 0;
    for (size_t i = 0; i < original_size; i++) {
        uint16_t *__restrict__ tree = m->o1[ctx];
        uint16_t *__restrict__ tree2 = m->o2[ctx2];
        int node = 1;
        for (int b = 7; b >= 0; b--) {
            uint32_t p1 = tree[node];
            uint32_t p2 = tree2[node];
            uint32_t p = (p1 + p2) >> 1;
            if (p < 1) p = 1; if (p > 4095) p = 4095;
            int bit = krc_dec_bit(&rc, p);
            if (bit) {
                tree[node] += (uint16_t)((4096 - p1) >> TURBO_RATE);
                tree2[node] += (uint16_t)((4096 - p2) >> TURBO_RATE);
            } else {
                tree[node] -= (uint16_t)(p1 >> TURBO_RATE);
                tree2[node] -= (uint16_t)(p2 >> TURBO_RATE);
            }
            node = node * 2 + bit;
        }
        int sym = node - 256;
        output[i] = (uint8_t)sym;
        ctx2 = ctx >> 7;
        ctx = sym;
    }

    free(m);
    return original_size;
}

/* --- Структуры и воркеры блочной обработки v62 --- */
typedef struct {
    const uint8_t *input;
    size_t input_size;
    uint8_t *output;
    size_t output_max;
    size_t original_size;   /* для декомпрессии */
    size_t result_size;
} KF62ThreadArg;

static void *kf62_compress_worker(void *arg) {
    KF62ThreadArg *a = (KF62ThreadArg *)arg;
    a->result_size = compress_formula_v62_block(
        a->input, a->input_size, a->output, a->output_max);
    return NULL;
}

/* v82: Fast BWT worker — без match hash table для экономии L3 cache */
static void *kf62_compress_worker_fast(void *arg) {
    KF62ThreadArg *a = (KF62ThreadArg *)arg;
    a->result_size = compress_formula_v62_block_fast(
        a->input, a->input_size, a->output, a->output_max);
    return NULL;
}

/* v85: Turbo worker — byte-level order-1 RC (10-30ms vs 200ms CM) */
static void *turbo_compress_worker(void *arg) {
    KF62ThreadArg *a = (KF62ThreadArg *)arg;
    a->result_size = compress_turbo_block(
        a->input, a->input_size, a->output, a->output_max);
    return NULL;
}

static void *kf62_decompress_worker(void *arg) {
    KF62ThreadArg *a = (KF62ThreadArg *)arg;
    a->result_size = decompress_formula_v62_block(
        a->input, a->input_size, a->output, a->output_max, a->original_size);
    return NULL;
}

/* v81: LZCM worker для параллельного тестирования BWT вариантов */
static void *kf62_lzcm_compress_worker(void *arg) {
    KF62ThreadArg *a = (KF62ThreadArg *)arg;
    a->result_size = compress_lzcm_v66_block(
        a->input, a->input_size, a->output, a->output_max);
    return NULL;
}

/* v81: Full LZCM worker (multi-block wrapper) для фонового сжатия */
static void *kf62_lzcm_full_worker(void *arg) {
    KF62ThreadArg *a = (KF62ThreadArg *)arg;
    a->result_size = compress_lzcm_v66(
        a->input, a->input_size, a->output, a->output_max);
    return NULL;
}

/* --- Основной компрессор v62 (блочный + потоковый) --- */
static size_t compress_formula_v62(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_max)
{
    if (input_size == 0 || output_max < 16) return 0;

    size_t nblocks = (input_size + KF62_BLOCK_SIZE - 1) / KF62_BLOCK_SIZE;
    if (nblocks > 65535) nblocks = 65535;

    size_t hdr_size = 2 + nblocks * 4;
    if (hdr_size >= output_max) return 0;

    uint8_t **bbufs = (uint8_t **)calloc(nblocks, sizeof(uint8_t*));
    size_t *bsizes  = (size_t *)calloc(nblocks, sizeof(size_t));
    size_t *bcsizes = (size_t *)calloc(nblocks, sizeof(size_t));
    if (!bbufs || !bsizes || !bcsizes) goto v62c_fail;

    for (size_t i = 0; i < nblocks; i++) {
        size_t off = i * KF62_BLOCK_SIZE;
        bsizes[i] = MIN((size_t)KF62_BLOCK_SIZE, input_size - off);
        bbufs[i] = (uint8_t *)malloc(bsizes[i] + 256);
        if (!bbufs[i]) goto v62c_fail;
    }

#if KF_USE_THREADS
    {
        size_t max_threads = kolibri_thread_count(nblocks);
        KF62ThreadArg *args = (KF62ThreadArg *)calloc(nblocks, sizeof(KF62ThreadArg));
        pthread_t *tids = (pthread_t *)calloc(nblocks, sizeof(pthread_t));
        if (!args || !tids) { free(args); free(tids); goto v62c_fail; }

        for (size_t start = 0; start < nblocks; start += max_threads) {
            size_t batch = MIN(max_threads, nblocks - start);
            for (size_t j = 0; j < batch; j++) {
                size_t idx = start + j;
                args[idx].input      = input + idx * KF62_BLOCK_SIZE;
                args[idx].input_size = bsizes[idx];
                args[idx].output     = bbufs[idx];
                args[idx].output_max = bsizes[idx] + 256;
                pthread_create(&tids[idx], NULL, kf62_compress_worker, &args[idx]);
            }
            for (size_t j = 0; j < batch; j++)
                pthread_join(tids[start + j], NULL);
            for (size_t j = 0; j < batch; j++)
                bcsizes[start + j] = args[start + j].result_size;
        }
        free(args); free(tids);
    }
#else
    for (size_t i = 0; i < nblocks; i++) {
        bcsizes[i] = compress_formula_v62_block(
            input + i * KF62_BLOCK_SIZE, bsizes[i],
            bbufs[i], bsizes[i] + 256);
    }
#endif

    /* Записываем заголовок блоков */
    output[0] = (uint8_t)(nblocks & 0xFF);
    output[1] = (uint8_t)((nblocks >> 8) & 0xFF);
    size_t pos = 2;
    for (size_t i = 0; i < nblocks; i++) {
        uint32_t cs = (uint32_t)bcsizes[i];
        memcpy(output + pos, &cs, 4);
        pos += 4;
    }

    /* Записываем данные блоков */
    for (size_t i = 0; i < nblocks; i++) {
        if (bcsizes[i] == 0) {
            /* Несжимаемый блок — raw */
            if (pos + bsizes[i] > output_max) goto v62c_fail;
            memcpy(output + pos, input + i * KF62_BLOCK_SIZE, bsizes[i]);
            pos += bsizes[i];
        } else {
            if (pos + bcsizes[i] > output_max) goto v62c_fail;
            memcpy(output + pos, bbufs[i], bcsizes[i]);
            pos += bcsizes[i];
        }
    }

    for (size_t i = 0; i < nblocks; i++) free(bbufs[i]);
    free(bbufs); free(bsizes); free(bcsizes);

    if (pos >= input_size) return 0;
    return pos;

v62c_fail:
    if (bbufs) { for (size_t i = 0; i < nblocks; i++) free(bbufs[i]); }
    free(bbufs); free(bsizes); free(bcsizes);
    return 0;
}

/* --- Основной декомпрессор v62 (блочный + потоковый) --- */
static size_t decompress_formula_v62(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_max,
    size_t original_size)
{
    if (original_size == 0 || original_size > output_max || input_size < 2) return 0;

    uint16_t nblocks = (uint16_t)input[0] | ((uint16_t)input[1] << 8);
    if (nblocks == 0 || input_size < 2 + (size_t)nblocks * 4) return 0;

    size_t hpos = 2;
    uint32_t *csizes = (uint32_t *)malloc(nblocks * sizeof(uint32_t));
    if (!csizes) return 0;
    for (uint16_t i = 0; i < nblocks; i++) {
        memcpy(&csizes[i], input + hpos, 4);
        hpos += 4;
    }

#if KF_USE_THREADS
    {
        /* Предрассчитываем смещения блоков */
        size_t *offsets = (size_t *)malloc((nblocks + 1) * sizeof(size_t));
        size_t *bsizes  = (size_t *)malloc(nblocks * sizeof(size_t));
        if (!offsets || !bsizes) { free(csizes); free(offsets); free(bsizes); return 0; }

        offsets[0] = hpos;
        for (uint16_t i = 0; i < nblocks; i++) {
            bsizes[i] = MIN((size_t)KF62_BLOCK_SIZE, original_size - (size_t)i * KF62_BLOCK_SIZE);
            offsets[i + 1] = offsets[i] + (csizes[i] == 0 ? bsizes[i] : csizes[i]);
        }

        size_t max_threads = kolibri_thread_count(nblocks);
        KF62ThreadArg *args = (KF62ThreadArg *)calloc(nblocks, sizeof(KF62ThreadArg));
        pthread_t *tids = (pthread_t *)calloc(nblocks, sizeof(pthread_t));
        if (!args || !tids) {
            free(csizes); free(offsets); free(bsizes);
            free(args); free(tids);
            return 0;
        }

        for (size_t start = 0; start < nblocks; start += max_threads) {
            size_t batch = MIN(max_threads, (size_t)nblocks - start);
            for (size_t j = 0; j < batch; j++) {
                size_t idx = start + j;
                size_t ooff = idx * KF62_BLOCK_SIZE;
                if (csizes[idx] == 0) {
                    /* Raw block — прямое копирование */
                    memcpy(output + ooff, input + offsets[idx], bsizes[idx]);
                    args[idx].result_size = bsizes[idx];
                } else {
                    args[idx].input         = input + offsets[idx];
                    args[idx].input_size    = csizes[idx];
                    args[idx].output        = output + ooff;
                    args[idx].output_max    = bsizes[idx];
                    args[idx].original_size = bsizes[idx];
                    pthread_create(&tids[idx], NULL, kf62_decompress_worker, &args[idx]);
                }
            }
            for (size_t j = 0; j < batch; j++) {
                size_t idx = start + j;
                if (csizes[idx] != 0)
                    pthread_join(tids[idx], NULL);
            }
            /* Проверяем результат */
            for (size_t j = 0; j < batch; j++) {
                size_t idx = start + j;
                if (csizes[idx] != 0 && args[idx].result_size != bsizes[idx]) {
                    free(csizes); free(offsets); free(bsizes);
                    free(args); free(tids);
                    return 0;
                }
            }
        }

        free(csizes); free(offsets); free(bsizes);
        free(args); free(tids);
        return original_size;
    }
#else
    {
        size_t pos = hpos;
        size_t out_pos = 0;
        for (uint16_t i = 0; i < nblocks; i++) {
            size_t bsize = MIN((size_t)KF62_BLOCK_SIZE, original_size - out_pos);
            if (csizes[i] == 0) {
                if (pos + bsize > input_size) { free(csizes); return 0; }
                memcpy(output + out_pos, input + pos, bsize);
                pos += bsize;
            } else {
                if (pos + csizes[i] > input_size) { free(csizes); return 0; }
                size_t dec = decompress_formula_v62_block(
                    input + pos, csizes[i],
                    output + out_pos, bsize, bsize);
                if (dec != bsize) { free(csizes); return 0; }
                pos += csizes[i];
            }
            out_pos += bsize;
        }
        free(csizes);
        return out_pos;
    }
#endif
}

int kolibri_compress(KolibriCompressor *comp,
                     const uint8_t *input,
                     size_t input_size,
                     uint8_t **output,
                     size_t *output_size,
                     KolibriCompressStats *stats) {
    if (!comp || !output || !output_size) {
        return -1;
    }
    /* Пустой ввод допустим — input может быть NULL при input_size==0 */
    if (!input && input_size > 0) {
        return -1;
    }

    double start_time = get_time_ms();

    /* === Пустые файлы: сохраняем только заголовок === */
    if (input_size == 0) {
        size_t header_size = sizeof(KolibriCompressHeader);
        uint8_t *out_buf = (uint8_t *)calloc(1, header_size);
        if (!out_buf) return -1;
        KolibriCompressHeader *hdr = (KolibriCompressHeader *)out_buf;
        hdr->magic = KOLIBRI_COMPRESS_MAGIC;
        hdr->version = (uint16_t)KOLIBRI_COMPRESS_VERSION;
        hdr->methods = 0;
        hdr->original_size = 0;
        hdr->checksum = kolibri_checksum(NULL, 0);
        *output = out_buf;
        *output_size = header_size;
        if (stats) {
            stats->original_size = 0;
            stats->compressed_size = header_size;
            stats->compression_ratio = 1.0;
            stats->checksum = hdr->checksum;
            stats->file_type = KOLIBRI_FILE_UNKNOWN;
            stats->methods_used = 0;
            stats->compression_time_ms = get_time_ms() - start_time;
            stats->decompression_time_ms = 0;
        }
        return 0;
    }

    /* ================================================================
     * v76: BLAZING mode — предельная скорость (>1 GB/s, <0.1ms/100KB)
     * ================================================================
     * Включается через API: KOLIBRI_COMPRESS_BLAZING
     * Минимальный overhead: 16KB хеш (стек), 1 malloc (выход), без CRC.
     * Rep-match для повторных дистанций. 8-байтное расширение.
     * Цель: побить lz4 по скорости при сравнимом ratio.
     * ================================================================ */
    {
        int blazing = (comp->methods & KOLIBRI_COMPRESS_BLAZING) ? 1 : 0;
        if (blazing) {
            size_t hdr_sz = sizeof(KolibriCompressHeader);
            size_t work_sz = input_size + 1024;
            /* v84: используем temp_buffer для LZ-кодирования (reuse между вызовами) */
            if (comp->temp_buffer_size < work_sz) {
                free(comp->temp_buffer);
                comp->temp_buffer = (uint8_t *)malloc(work_sz);
                if (!comp->temp_buffer) { comp->temp_buffer_size = 0; return -1; }
                comp->temp_buffer_size = work_sz;
            }

            size_t lz_sz = lz_blazing_v76_encode(input, input_size,
                                                   comp->temp_buffer, work_sz);
            uint32_t methods = 0;
            size_t final_sz;
            if (lz_sz > 0 && lz_sz < input_size) {
                final_sz = lz_sz;
                methods = KOLIBRI_COMPRESS_LZ77;
            } else {
                final_sz = input_size;
            }

            /* Аллоцируем точный размер выхода (меньше, чем input_size+1024) */
            size_t total_out = hdr_sz + final_sz;
            uint8_t *obuf = (uint8_t *)malloc(total_out);
            if (!obuf) return -1;

            KolibriCompressHeader *h = (KolibriCompressHeader *)obuf;
            h->magic = KOLIBRI_COMPRESS_MAGIC;
            h->version = (uint16_t)KOLIBRI_COMPRESS_VERSION;
            h->methods = (uint16_t)methods;
            h->original_size = (uint32_t)input_size;
            h->checksum = 0;  /* blazing: skip CRC32 for speed */

            if (methods) {
                memcpy(obuf + hdr_sz, comp->temp_buffer, final_sz);
            } else {
                memcpy(obuf + hdr_sz, input, final_sz);
            }

            *output = obuf;
            *output_size = total_out;

            if (stats) {
                stats->original_size = input_size;
                stats->compressed_size = *output_size;
                stats->compression_ratio = (double)input_size / (double)*output_size;
                stats->checksum = 0;
                stats->file_type = KOLIBRI_FILE_UNKNOWN;
                stats->methods_used = methods;
                stats->compression_time_ms = get_time_ms() - start_time;
                stats->decompression_time_ms = 0;
            }
            return 0;
        }
    }

    /* v75: TURBO mode — быстрый LZ-only (без CM/Formula).
     * Включается через API: KOLIBRI_COMPRESS_TURBO, или через env: KOLIBRI_TURBO=1
     * FAST-PATH: пропускаем detect_file_type/token/formula/LZCM/BWT целиком.
     * Единственный malloc — выходной буфер, LZ кодирует прямо в него. */
    {
        int turbo = (comp->methods & KOLIBRI_COMPRESS_TURBO) ? 1 : 0;
        if (turbo) {
            size_t hdr_sz = sizeof(KolibriCompressHeader);
            size_t max_out = hdr_sz + input_size + 1024;
            uint8_t *obuf = (uint8_t *)malloc(max_out);
            if (!obuf) return -1;

            uint8_t *payload = obuf + hdr_sz;
            size_t lz_sz = lz_ultrafast_encode(input, input_size,
                                                payload, input_size + 1024);
            uint32_t methods = 0;
            size_t final_sz;
            if (lz_sz > 0 && lz_sz < input_size) {
                final_sz = lz_sz;
                methods = KOLIBRI_COMPRESS_LZ77;
            } else {
                memcpy(payload, input, input_size);
                final_sz = input_size;
            }

            KolibriCompressHeader *h = (KolibriCompressHeader *)obuf;
            h->magic = KOLIBRI_COMPRESS_MAGIC;
            h->version = (uint16_t)KOLIBRI_COMPRESS_VERSION;
            h->methods = (uint16_t)methods;
            h->original_size = (uint32_t)input_size;
            h->checksum = 0;  /* turbo: skip CRC32 for speed */

            *output = obuf;
            *output_size = hdr_sz + final_sz;

            if (stats) {
                stats->original_size = input_size;
                stats->compressed_size = *output_size;
                stats->compression_ratio = (double)input_size / (double)*output_size;
                stats->checksum = 0;
                stats->file_type = KOLIBRI_FILE_UNKNOWN;
                stats->methods_used = methods;
                stats->compression_time_ms = get_time_ms() - start_time;
                stats->decompression_time_ms = 0;
            }
            return 0;
        }
    }

    /* Detect file type */
    KolibriFileType file_type = kolibri_detect_file_type(input, input_size);

    /* ============================================================================
     * v81: FAST BWT PATH — для текстовых файлов прямой путь к лучшему варианту
     * (BWT → MTF → Zero-RLE → CM). Пропускает Token, LZ, LZCM, варианты A/B/D.
     * Для файлов ≥ 100KB вариант C (MTF+RLE+CM) побеждает всегда.
     * Для файлов < 100KB пробуем A+C (2 варианта вместо 6 проходов).
     * Ускорение ~5-7× при идентичном результате сжатия.
     * ============================================================================ */
    #define KOLIBRI_FAST_BWT_THRESHOLD 200 /* минимальный размер для fast path */
    if (file_type == KOLIBRI_FILE_TEXT
        && input_size >= KOLIBRI_FAST_BWT_THRESHOLD
        && input_size <= (8 * 1024 * 1024)) {

        saidx_t fast_pidx = 0;
        uint8_t *fast_bwt = (uint8_t *)malloc(input_size);
        if (fast_bwt && bw_transform(input, fast_bwt, NULL,
                                      (saidx_t)input_size, &fast_pidx) == 0) {
#if KOLIBRI_TRACE_BWT
            double t_bwt = get_time_ms() - start_time;
#endif
            /* MTF */
            uint8_t *fast_mtf = (uint8_t *)malloc(input_size);
            if (fast_mtf) {
                memcpy(fast_mtf, fast_bwt, input_size);
                kolibri_mtf_forward(fast_mtf, input_size);

                /* Zero-RLE */
                uint8_t *fast_rle = (uint8_t *)malloc(input_size * 2);
                size_t rle_sz = 0;
                if (fast_rle) {
                    rle_sz = kolibri_zero_rle_encode(
                        fast_mtf, input_size, fast_rle, input_size * 2);
                }

                /* Определяем кандидатов:
                 * - Для файлов ≥ 100KB: только C (MTF+RLE+CM)
                 * - Для < 100KB: A (MTF+CM) и C (MTF+RLE+CM) параллельно */
                int try_a = (input_size < 100000) ? 1 : 0;
                int rle_ok = (fast_rle && rle_sz > 0 && rle_sz < input_size);

                KF62ThreadArg fast_args[2]; /* [0]=A, [1]=C */
                memset(fast_args, 0, sizeof(fast_args));

                /* Вариант A: MTF + CM (только для маленьких файлов) */
                if (try_a) {
                    fast_args[0].input = fast_mtf;
                    fast_args[0].input_size = input_size;
                    fast_args[0].output = (uint8_t *)malloc(input_size + 1024);
                    fast_args[0].output_max = input_size + 1024;
                }

                /* Вариант C: RLE(MTF) + CM */
                if (rle_ok) {
                    fast_args[1].input = fast_rle;
                    fast_args[1].input_size = rle_sz;
                    fast_args[1].output = (uint8_t *)malloc(rle_sz + 1024);
                    fast_args[1].output_max = rle_sz + 1024;
                }

#if KF_USE_THREADS
                {
                    pthread_t ftids[2];
                    int flaunched[2] = {0};
                    if (try_a && fast_args[0].output) {
                        /* v83: full worker для fmt=2 — decompressor uses
                         * decompress_formula_v62_block (with match model),
                         * so compressor must also use match model */
                        pthread_create(&ftids[0], NULL,
                                       kf62_compress_worker, &fast_args[0]);
                        flaunched[0] = 1;
                    }
                    if (rle_ok && fast_args[1].output) {
                        /* v85: turbo worker — byte-level order-1 RC (10-30ms vs 200ms CM) */
                        pthread_create(&ftids[1], NULL,
                                       turbo_compress_worker, &fast_args[1]);
                        flaunched[1] = 1;
                    }
                    for (int fi = 0; fi < 2; fi++)
                        if (flaunched[fi]) pthread_join(ftids[fi], NULL);
                }
#else
                if (try_a && fast_args[0].output)
                    kf62_compress_worker(&fast_args[0]);  /* v83: full for fmt=2 */
                if (rle_ok && fast_args[1].output)
                    turbo_compress_worker(&fast_args[1]);  /* v85: turbo */
#endif

                /* Выбираем лучший результат */
                size_t best_sz = 0;
                int best_fmt = -1; /* -1=none, 2=A(MTF), 5=C(MTF+RLE) */
                uint8_t *best_data = NULL;
                size_t best_data_sz = 0;

                /* A: MTF+CM */
                if (try_a && fast_args[0].output && fast_args[0].result_size > 0) {
                    size_t total = 10 + fast_args[0].result_size; /* BWT_HDR + CM */
#if KOLIBRI_TRACE_BWT
                    fprintf(stderr, "  [FAST-A] MTF+CM: %zu\n", total);
#endif
                    if (total < input_size) {
                        best_sz = total;
                        best_fmt = 2; /* KOLIBRI_BWT_FMT_MTF */
                        best_data = fast_args[0].output;
                        best_data_sz = fast_args[0].result_size;
                    }
                }

                /* C: MTF+RLE+TURBO */
                if (rle_ok && fast_args[1].output && fast_args[1].result_size > 0) {
                    size_t total = 10 + 3 + fast_args[1].result_size;
#if KOLIBRI_TRACE_BWT
                    fprintf(stderr, "  [FAST-T] MTF+RLE+TURBO: %zu\n", total);
#endif
                    if (total < input_size && (best_sz == 0 || total < best_sz)) {
                        best_sz = total;
                        best_fmt = 7; /* KOLIBRI_BWT_FMT_MTF_RLE_TURBO */
                        best_data = fast_args[1].output;
                        best_data_sz = fast_args[1].result_size;
                    }
                }

                if (best_fmt >= 0 && best_sz > 0) {
                    /* Собираем выход */
                    uint8_t *fast_out = (uint8_t *)malloc(best_sz);
                    if (fast_out) {
                        uint16_t bm = 0x424B; /* KOLIBRI_BWT_MAGIC */
                        memcpy(fast_out, &bm, 2);
                        fast_out[2] = (uint8_t)(input_size & 0xFF);
                        fast_out[3] = (uint8_t)((input_size >> 8) & 0xFF);
                        fast_out[4] = (uint8_t)((input_size >> 16) & 0xFF);
                        uint32_t pidx32 = (uint32_t)fast_pidx;
                        memcpy(fast_out + 5, &pidx32, 4);
                        fast_out[9] = (uint8_t)best_fmt;

                        if (best_fmt == 5 || best_fmt == 7) {
                            /* RLE: [rle_size(3)] + CM data */
                            fast_out[10] = (uint8_t)(rle_sz & 0xFF);
                            fast_out[11] = (uint8_t)((rle_sz >> 8) & 0xFF);
                            fast_out[12] = (uint8_t)((rle_sz >> 16) & 0xFF);
                            memcpy(fast_out + 13, best_data, best_data_sz);
                        } else {
                            /* fmt=2: MTF CM data directly after header */
                            memcpy(fast_out + 10, best_data, best_data_sz);
                        }

                        *output = fast_out;
                        *output_size = best_sz;
                        if (stats) {
                            stats->original_size = input_size;
                            stats->compressed_size = best_sz;
                            stats->compression_ratio = (double)input_size / (double)best_sz;
                            stats->checksum = kolibri_checksum(input, input_size);
                            stats->file_type = file_type;
                            stats->methods_used = KOLIBRI_COMPRESS_FORMULA;
                            stats->compression_time_ms = get_time_ms() - start_time;
                            stats->decompression_time_ms = 0;
                        }
#if KOLIBRI_TRACE_BWT
                        fprintf(stderr, "[FAST_BWT] %zu -> %zu (%.3fx) fmt=%d"
                                " bwt=%.0fms total=%.0fms\n",
                                input_size, best_sz,
                                (double)input_size / best_sz, best_fmt,
                                t_bwt,
                                stats ? stats->compression_time_ms : 0.0);
#endif

                        /* Cleanup и выход */
                        free(fast_args[0].output);
                        free(fast_args[1].output);
                        free(fast_rle);
                        free(fast_mtf);
                        free(fast_bwt);
                        return 0;
                    }
                }

                /* Fast path не сработал — очистка и fallback */
                free(fast_args[0].output);
                free(fast_args[1].output);
                free(fast_rle);
                free(fast_mtf);
            }
        }
        free(fast_bwt);
        /* Fallback: продолжаем со стандартным путём */
    }

    /* Allocate output buffer */
    size_t max_output = sizeof(KolibriCompressHeader) + input_size + 16384;
    uint8_t *out_buf = (uint8_t *)malloc(max_output);
    if (!out_buf) return -1;

    size_t header_size = sizeof(KolibriCompressHeader);
    uint8_t *compressed_data = out_buf + header_size;

    size_t compressed_size = input_size;
    uint32_t methods_used = 0;

    /* === KOLIBRI v52 HYBRID: Token LZ-lite + Formula ===
     * Этап 1: LZ-lite убирает длинные повторы (дёшево)
     * Этап 2: Formula кодирует остаток (адаптивно) */

    /* Буферы */
    uint8_t *lz_buf = (uint8_t *)malloc(input_size + 1024);
    uint8_t *temp = (uint8_t *)malloc(input_size + 1024);
    if (!lz_buf || !temp) { free(lz_buf); free(temp); free(out_buf); return -1; }

    /* Token-level text stream (v52) */
    TokenDict tdict = {0};
    uint8_t *token_buf = NULL;
    size_t token_size = 0;
    int token_used = 0;
    size_t token_dict_bytes = 0;

    const uint8_t *formula_input = input;
    size_t formula_input_size = input_size;
    int lz_used = 0;

    int turbo = 0;  /* quality path — turbo уже вышел выше */

    if (file_type == KOLIBRI_FILE_TEXT && !turbo) {
        if (token_dict_build(&tdict, input, input_size)) {
            token_buf = (uint8_t *)malloc(input_size * 2 + 16);
            if (token_buf) {
                token_size = token_encode_text(input, input_size,
                                               token_buf, input_size * 2 + 16,
                                               &tdict);
                if (token_size > 0 && token_size < input_size) {
                    token_used = 1;
                    formula_input = token_buf;
                    formula_input_size = token_size;
                } else {
                    free(token_buf);
                    token_buf = NULL;
                    token_dict_free(&tdict);
                }
            } else {
                token_dict_free(&tdict);
            }
        }
    }

    /* Этап 1: LZ — ultrafast (0 цепочек, 64K хеш) или lite (цепочки) */
    size_t lz_size = turbo
        ? lz_ultrafast_encode(formula_input, formula_input_size,
                              lz_buf, formula_input_size + 1024)
        : lz_lite_encode(formula_input, formula_input_size,
                         lz_buf, formula_input_size + 1024);
    if (lz_size > 0 && lz_size < formula_input_size) {
        formula_input = lz_buf;
        formula_input_size = lz_size;
        lz_used = 1;
    }

    /* v81: Спекулятивный запуск LZCM в фоновом потоке параллельно с CM_trad.
     * LZCM работает на оригинальном input (не formula_input), поэтому
     * полностью независим от tokenizации/LZ. */
    #define KOLIBRI_LZCM_BWT_MAX_INPUT_EARLY (8 * 1024 * 1024)
    uint8_t *lzcm_buf_bg = NULL;
    int lzcm_bg_started = 0;
#if KF_USE_THREADS
    KF62ThreadArg lzcm_bg_arg;
    pthread_t lzcm_bg_tid;
    if (!turbo && input_size >= 64 && input_size <= KOLIBRI_LZCM_BWT_MAX_INPUT_EARLY) {
        lzcm_buf_bg = (uint8_t *)malloc(input_size + 1024);
        if (lzcm_buf_bg) {
            memset(&lzcm_bg_arg, 0, sizeof(lzcm_bg_arg));
            lzcm_bg_arg.input = input;
            lzcm_bg_arg.input_size = input_size;
            lzcm_bg_arg.output = lzcm_buf_bg;
            lzcm_bg_arg.output_max = input_size + 1024;
            pthread_create(&lzcm_bg_tid, NULL, kf62_lzcm_full_worker, &lzcm_bg_arg);
            lzcm_bg_started = 1;
        }
    }
#endif

    /* Этап 2: Формульное сжатие (QUALITY only — в TURBO пропускаем) */
    size_t formula_size = 0;
    if (!turbo) {
        formula_size = compress_formula_v62(formula_input, formula_input_size,
                                           temp, input_size + 1024);
    }
    if (formula_size > 0 && formula_size < formula_input_size) {
        size_t prefix_size = 0;
        uint8_t *payload = compressed_data;
        methods_used = KOLIBRI_COMPRESS_FORMULA;
        if (token_used) {
            token_dict_bytes = token_dict_write(&tdict, payload,
                                                max_output - header_size);
            if (token_dict_bytes == 0) {
                methods_used = 0;
                compressed_size = input_size;
                memcpy(compressed_data, input, input_size);
                goto compress_done;
            } else {
                payload += token_dict_bytes;
                uint32_t token_intermediate = (uint32_t)token_size;
                memcpy(payload, &token_intermediate, 4);
                payload += 4;
                prefix_size = token_dict_bytes + 4;
                methods_used |= KOLIBRI_COMPRESS_TOKEN;
            }
        }
        if (lz_used) {
            methods_used |= KOLIBRI_COMPRESS_LZ77;
            uint32_t lz_intermediate = (uint32_t)formula_input_size;
            memcpy(payload, &lz_intermediate, 4);
            payload += 4;
            prefix_size += 4;
        }
        memcpy(payload, temp, formula_size);
        compressed_size = prefix_size + formula_size;
        if (compressed_size >= input_size) {
            methods_used = 0;
            compressed_size = input_size;
            memcpy(compressed_data, input, input_size);
        }
    } else if (lz_used) {
        size_t prefix_size = 0;
        uint8_t *payload = compressed_data;
        methods_used = KOLIBRI_COMPRESS_LZ77;
        if (token_used) {
            token_dict_bytes = token_dict_write(&tdict, payload,
                                                max_output - header_size);
            if (token_dict_bytes == 0) {
                methods_used = 0;
                compressed_size = input_size;
                memcpy(compressed_data, input, input_size);
                goto compress_done;
            } else {
                payload += token_dict_bytes;
                uint32_t token_intermediate = (uint32_t)token_size;
                memcpy(payload, &token_intermediate, 4);
                payload += 4;
                prefix_size = token_dict_bytes + 4;
                methods_used |= KOLIBRI_COMPRESS_TOKEN;
            }
        }
        memcpy(payload, lz_buf, lz_size);
        compressed_size = prefix_size + lz_size;
        if (compressed_size >= input_size) {
            methods_used = 0;
            compressed_size = input_size;
            memcpy(compressed_data, input, input_size);
        }
    } else {
        /* Данные несжимаемы — сохраняем как есть */
        compressed_size = input_size;
        methods_used = 0;
        memcpy(compressed_data, input, input_size);
    }

    /* ================================================================
     * v66: LZCM unified encoder — LZ matching + CM для литералов.
     * Архитектура уровня LZMA: значительно превосходит раздельный LZ → CM.
     * v66b: Минимальный 6-байтный заголовок для single-block LZCM.
     * ================================================================ */

    /* v70: Move-to-Front преобразование для BWT выхода.
     * После BWT данные содержат длинные серии одинаковых символов.
     * MTF конвертирует их в малые числа (много нулей), которые
     * CM модель предсказывает значительно лучше. */

    /* v70: BWT+LZCM header constants */
    #define KOLIBRI_BWT_MAGIC 0x424B  /* "KB" */
    #define KOLIBRI_BWT_HDR_SIZE 10   /* magic(2) + orig_size(3) + pidx(4) + fmt(1) */
    #define KOLIBRI_BWT_FMT_LZCM 0   /* BWT + LZCM (legacy) */
    #define KOLIBRI_BWT_FMT_CM   1   /* BWT + pure CM (no LZ) */
    #define KOLIBRI_BWT_FMT_MTF  2   /* BWT + MTF + CM (best for text) */
    #define KOLIBRI_BWT_FMT_DELTA 3  /* BWT + Delta + CM */
    #define KOLIBRI_BWT_FMT_MTF_RLE 5 /* v77: BWT + MTF + Zero-RLE + CM */

    /* v75c: Size guard — LZCM/BWT заголовки используют 3-байтный original_size
     * (max 16 777 215 = 16 МБ). Для данных > 8 МБ пропускаем этот блок
     * и используем традиционный путь с 4-байтным заголовком.
     * Также для очень больших файлов экономим ~25× памяти. */
    #define KOLIBRI_LZCM_BWT_MAX_INPUT (8 * 1024 * 1024)  /* 8 МБ */

    if (!turbo && compressed_size > 64 && input_size <= KOLIBRI_LZCM_BWT_MAX_INPUT) {
        /* v81: Используем результат фонового потока LZCM если он был запущен */
        uint8_t *lzcm_buf = NULL;
        size_t lzcm_size = 0;
#if KF_USE_THREADS
        if (lzcm_bg_started) {
            pthread_join(lzcm_bg_tid, NULL);
            lzcm_buf = lzcm_buf_bg;
            lzcm_size = lzcm_bg_arg.result_size;
            lzcm_buf_bg = NULL;  /* ownership transferred */
            lzcm_bg_started = 0;
        } else
#endif
        {
            lzcm_buf = (uint8_t *)malloc(input_size + 1024);
            if (lzcm_buf) {
                lzcm_size = compress_lzcm_v66(input, input_size,
                                               lzcm_buf, input_size + 1024);
            }
        }

        if (lzcm_buf) {

            /* v70: Определяем лучший результат — LZCM или BWT+CM или BWT+LZCM */
            size_t best_total = 0;
            int best_is_bwt = 0;
            int bwt_fmt = KOLIBRI_BWT_FMT_LZCM;  /* какой формат BWT победил */
            uint8_t *bwt_lzcm_data = NULL;
            size_t bwt_lzcm_data_size = 0;
            saidx_t bwt_pidx = 0;

            if (lzcm_size > 1 && lzcm_buf[0] == 0x00) {
                best_total = KOLIBRI_LZCM_HDR_SIZE + (lzcm_size - 1);
            }
#if KOLIBRI_TRACE_BWT
            fprintf(stderr, "[BWT_DEBUG] LZCM_direct=%zu trad=%zu\n",
                    best_total, header_size + compressed_size);
#endif

            /* v72: BWT preprocessing — пробуем MTF+CM и MTF+LZCM,
             * выбираем лучший результат. LZCM на MTF потоке эффективно
             * кодирует нулевые серии как LZ матчи + CM для остального. */
            #define KOLIBRI_BWT_FMT_MTF_LZCM 4  /* v72: BWT + MTF + LZCM */
            if (file_type == KOLIBRI_FILE_TEXT && input_size >= 64) {
                uint8_t *bwt_buf = (uint8_t *)malloc(input_size);
                if (bwt_buf) {
                    int bwt_ok = (bw_transform(input, bwt_buf, NULL,
                                               (saidx_t)input_size, &bwt_pidx) == 0);
                    if (bwt_ok) {
                        /* Шаг 1: BWT + MTF */
                        uint8_t *mtf_buf = (uint8_t *)malloc(input_size);
                        if (mtf_buf) {
                            memcpy(mtf_buf, bwt_buf, input_size);
                            kolibri_mtf_forward(mtf_buf, input_size);

                            /* v81: Preprocessing (BWT+MTF общий, RLE для C, multi-block для D)
                             * затем параллельный CM для всех вариантов.
                             * Формат-константы определяем один раз */
                            #define KOLIBRI_BWT_FMT_MTF_LZCM 4
                            #define KOLIBRI_BWT_FMT_MTF_RLE_MB 6
                            #define KOLIBRI_BWT_MB_BLOCK 131072u  /* 128KB */

                            /* --- Preprocessing для варианта C: RLE --- */
                            uint8_t *rle_buf_c = (uint8_t *)malloc(input_size * 2);
                            size_t rle_size_c = 0;
                            if (rle_buf_c) {
                                rle_size_c = kolibri_zero_rle_encode(
                                    mtf_buf, input_size, rle_buf_c, input_size * 2);
                                if (rle_size_c == 0 || rle_size_c >= input_size) {
                                    free(rle_buf_c); rle_buf_c = NULL;
                                }
                            }

                            /* --- Preprocessing для варианта D: multi-block BWT+MTF+RLE --- */
                            uint8_t *rle_total_d = NULL;
                            size_t total_rle_d = 0;
                            size_t nblocks_d = 0;
                            size_t *mb_bs_d = NULL;
                            saidx_t *mb_pidx_d = NULL;
                            size_t *mb_rle_sz_d = NULL;
                            int d_prep_ok = 0;

                            if (input_size >= 200000) {
                                nblocks_d = (input_size + KOLIBRI_BWT_MB_BLOCK - 1)
                                            / KOLIBRI_BWT_MB_BLOCK;
                                if (nblocks_d >= 2 && nblocks_d <= 255) {
                                    mb_bs_d = (size_t *)calloc(nblocks_d, sizeof(size_t));
                                    mb_pidx_d = (saidx_t *)calloc(nblocks_d, sizeof(saidx_t));
                                    mb_rle_sz_d = (size_t *)calloc(nblocks_d, sizeof(size_t));
                                    uint8_t **mb_rle_d = (uint8_t **)calloc(nblocks_d, sizeof(uint8_t *));

                                    if (mb_bs_d && mb_pidx_d && mb_rle_sz_d && mb_rle_d) {
                                        int mb_ok = 1;
                                        size_t offset = 0;
                                        for (size_t bi = 0; bi < nblocks_d && mb_ok; bi++) {
                                            size_t bs = (bi < nblocks_d - 1)
                                                ? KOLIBRI_BWT_MB_BLOCK
                                                : (input_size - offset);
                                            mb_bs_d[bi] = bs;
                                            uint8_t *bwt_b = (uint8_t *)malloc(bs);
                                            if (!bwt_b) { mb_ok = 0; break; }
                                            if (bw_transform(input + offset, bwt_b, NULL,
                                                             (saidx_t)bs, &mb_pidx_d[bi]) != 0) {
                                                free(bwt_b); mb_ok = 0; break;
                                            }
                                            kolibri_mtf_forward(bwt_b, bs);
                                            uint8_t *rle_b = (uint8_t *)malloc(bs * 2);
                                            if (!rle_b) { free(bwt_b); mb_ok = 0; break; }
                                            size_t rle_sz = kolibri_zero_rle_encode(
                                                bwt_b, bs, rle_b, bs * 2);
                                            free(bwt_b);
                                            if (rle_sz == 0 || rle_sz >= bs) {
                                                free(rle_b); mb_ok = 0; break;
                                            }
                                            mb_rle_d[bi] = rle_b;
                                            mb_rle_sz_d[bi] = rle_sz;
                                            total_rle_d += rle_sz;
                                            offset += bs;
                                        }
                                        if (mb_ok && total_rle_d > 0) {
                                            rle_total_d = (uint8_t *)malloc(total_rle_d);
                                            if (rle_total_d) {
                                                size_t roff = 0;
                                                for (size_t bi = 0; bi < nblocks_d; bi++) {
                                                    memcpy(rle_total_d + roff,
                                                           mb_rle_d[bi], mb_rle_sz_d[bi]);
                                                    roff += mb_rle_sz_d[bi];
                                                }
                                                d_prep_ok = 1;
                                            }
                                        }
                                        for (size_t bi = 0; bi < nblocks_d; bi++)
                                            free(mb_rle_d[bi]);
                                    }
                                    free(mb_rle_d);
                                }
                            }

                            /* --- CM: определяем аргументы для всех вариантов --- */
                            KF62ThreadArg var_args[4];
                            memset(var_args, 0, sizeof(var_args));

                            /* A: MTF + CM */
                            var_args[0].input = mtf_buf;
                            var_args[0].input_size = input_size;
                            var_args[0].output = (uint8_t *)malloc(input_size + 1024);
                            var_args[0].output_max = input_size + 1024;

                            /* B: MTF + LZCM */
                            var_args[1].input = mtf_buf;
                            var_args[1].input_size = input_size;
                            var_args[1].output = (uint8_t *)malloc(input_size + 1024);
                            var_args[1].output_max = input_size + 1024;

                            /* C: RLE(MTF) + CM */
                            if (rle_buf_c) {
                                var_args[2].input = rle_buf_c;
                                var_args[2].input_size = rle_size_c;
                                var_args[2].output = (uint8_t *)malloc(rle_size_c + 1024);
                                var_args[2].output_max = rle_size_c + 1024;
                            }

                            /* D: concatenated multi-block RLE + CM */
                            if (d_prep_ok) {
                                var_args[3].input = rle_total_d;
                                var_args[3].input_size = total_rle_d;
                                var_args[3].output = (uint8_t *)malloc(total_rle_d + 1024);
                                var_args[3].output_max = total_rle_d + 1024;
                            }


#if KF_USE_THREADS
                            /* v81: Параллельный CM — все 4 варианта одновременно */
                            {
                                pthread_t var_tids[4];
                                int launched[4] = {0};
                                if (var_args[0].output) {
                                    pthread_create(&var_tids[0], NULL,
                                                   kf62_compress_worker, &var_args[0]);
                                    launched[0] = 1;
                                }
                                if (var_args[1].output) {
                                    pthread_create(&var_tids[1], NULL,
                                                   kf62_lzcm_compress_worker, &var_args[1]);
                                    launched[1] = 1;
                                }
                                if (rle_buf_c && var_args[2].output) {
                                    /* v82: fast worker для BWT+RLE (без match model) */
                                    pthread_create(&var_tids[2], NULL,
                                                   kf62_compress_worker_fast, &var_args[2]);
                                    launched[2] = 1;
                                }
                                if (d_prep_ok && var_args[3].output) {
                                    pthread_create(&var_tids[3], NULL,
                                                   kf62_compress_worker, &var_args[3]);
                                    launched[3] = 1;
                                }
                                for (int vi = 0; vi < 4; vi++)
                                    if (launched[vi]) pthread_join(var_tids[vi], NULL);
                            }
#else
                            /* Последовательная обработка */
                            if (var_args[0].output)
                                kf62_compress_worker(&var_args[0]);
                            if (var_args[1].output)
                                kf62_lzcm_compress_worker(&var_args[1]);
                            if (rle_buf_c && var_args[2].output)
                                kf62_compress_worker_fast(&var_args[2]);  /* v82: fast */
                            if (d_prep_ok && var_args[3].output)
                                kf62_compress_worker(&var_args[3]);
#endif

                            /* --- Выбор лучшего результата из A/B/C/D --- */

                            /* A: MTF + CM */
                            if (var_args[0].output && var_args[0].result_size > 0) {
                                size_t bwt_total = KOLIBRI_BWT_HDR_SIZE + var_args[0].result_size;
#if KOLIBRI_TRACE_BWT
                                fprintf(stderr, "  [A] MTF+CM: %zu\n", bwt_total);
#endif
                                if (best_total == 0 || bwt_total < best_total) {
                                    best_total = bwt_total;
                                    best_is_bwt = 1;
                                    bwt_fmt = KOLIBRI_BWT_FMT_MTF;
                                    free(bwt_lzcm_data);
                                    bwt_lzcm_data = var_args[0].output;
                                    bwt_lzcm_data_size = var_args[0].result_size;
                                    var_args[0].output = NULL;
                                }
                            }

                            /* B: MTF + LZCM */
                            if (var_args[1].output && var_args[1].result_size > 0) {
                                size_t bwt_total = KOLIBRI_BWT_HDR_SIZE + var_args[1].result_size;
#if KOLIBRI_TRACE_BWT
                                fprintf(stderr, "  [B] MTF+LZCM: %zu\n", bwt_total);
#endif
                                if (best_total == 0 || bwt_total < best_total) {
                                    best_total = bwt_total;
                                    best_is_bwt = 1;
                                    bwt_fmt = KOLIBRI_BWT_FMT_MTF_LZCM;
                                    free(bwt_lzcm_data);
                                    bwt_lzcm_data = var_args[1].output;
                                    bwt_lzcm_data_size = var_args[1].result_size;
                                    var_args[1].output = NULL;
                                }
                            }

                            /* C: MTF + RLE + CM */
                            if (rle_buf_c && var_args[2].output &&
                                var_args[2].result_size > 0) {
                                size_t cm_rle_size = var_args[2].result_size;
                                size_t bwt_total = KOLIBRI_BWT_HDR_SIZE + 3 + cm_rle_size;
#if KOLIBRI_TRACE_BWT
                                fprintf(stderr, "  [C] MTF+RLE+CM: %zu (rle_in=%zu rle_out=%zu cm=%zu)\n",
                                        bwt_total, input_size, rle_size_c, cm_rle_size);
#endif
                                if (best_total == 0 || bwt_total < best_total) {
                                    best_total = bwt_total;
                                    best_is_bwt = 1;
                                    bwt_fmt = KOLIBRI_BWT_FMT_MTF_RLE;
                                    free(bwt_lzcm_data);
                                    bwt_lzcm_data = (uint8_t *)malloc(3 + cm_rle_size);
                                    if (bwt_lzcm_data) {
                                        bwt_lzcm_data[0] = (uint8_t)(rle_size_c & 0xFF);
                                        bwt_lzcm_data[1] = (uint8_t)((rle_size_c >> 8) & 0xFF);
                                        bwt_lzcm_data[2] = (uint8_t)((rle_size_c >> 16) & 0xFF);
                                        memcpy(bwt_lzcm_data + 3, var_args[2].output, cm_rle_size);
                                        bwt_lzcm_data_size = 3 + cm_rle_size;
                                    }
                                }
                            }

                            /* D: Multi-block MTF+RLE+CM */
                            if (d_prep_ok && var_args[3].output &&
                                var_args[3].result_size > 0) {
                                size_t cm_sz = var_args[3].result_size;
                                size_t meta_sz = 1 + nblocks_d * 10;
                                size_t bwt_total = KOLIBRI_BWT_HDR_SIZE + meta_sz + cm_sz;
#if KOLIBRI_TRACE_BWT
                                fprintf(stderr, "  [D] MB(%zu×128K) MTF+RLE+CM: %zu (total_rle=%zu cm=%zu)\n",
                                        nblocks_d, bwt_total, total_rle_d, cm_sz);
#endif
                                if (best_total == 0 || bwt_total < best_total) {
                                    best_total = bwt_total;
                                    best_is_bwt = 1;
                                    bwt_fmt = KOLIBRI_BWT_FMT_MTF_RLE_MB;
                                    bwt_pidx = 0;
                                    free(bwt_lzcm_data);
                                    bwt_lzcm_data = (uint8_t *)malloc(meta_sz + cm_sz);
                                    if (bwt_lzcm_data) {
                                        bwt_lzcm_data[0] = (uint8_t)nblocks_d;
                                        for (size_t bi = 0; bi < nblocks_d; bi++) {
                                            size_t mi = 1 + bi * 10;
                                            bwt_lzcm_data[mi  ] = (uint8_t)(mb_bs_d[bi] & 0xFF);
                                            bwt_lzcm_data[mi+1] = (uint8_t)((mb_bs_d[bi] >> 8) & 0xFF);
                                            bwt_lzcm_data[mi+2] = (uint8_t)((mb_bs_d[bi] >> 16) & 0xFF);
                                            uint32_t pidx32 = (uint32_t)mb_pidx_d[bi];
                                            memcpy(bwt_lzcm_data + mi + 3, &pidx32, 4);
                                            bwt_lzcm_data[mi+7] = (uint8_t)(mb_rle_sz_d[bi] & 0xFF);
                                            bwt_lzcm_data[mi+8] = (uint8_t)((mb_rle_sz_d[bi] >> 8) & 0xFF);
                                            bwt_lzcm_data[mi+9] = (uint8_t)((mb_rle_sz_d[bi] >> 16) & 0xFF);
                                        }
                                        memcpy(bwt_lzcm_data + meta_sz, var_args[3].output, cm_sz);
                                        bwt_lzcm_data_size = meta_sz + cm_sz;
                                    }
                                }
                            }

                            /* Cleanup всех вариантов */
                            free(var_args[0].output);
                            free(var_args[1].output);
                            free(var_args[2].output);
                            free(var_args[3].output);
                            free(rle_buf_c);
                            free(rle_total_d);
                            free(mb_bs_d); free(mb_pidx_d); free(mb_rle_sz_d);

#if KOLIBRI_TRACE_BWT
                            fprintf(stderr, "[BWT_DEBUG] input=%zu best=%zu fmt=%d\n",
                                    input_size, best_total, bwt_fmt);
#endif

                            free(mtf_buf);
                        }
                    }
                    free(bwt_buf);
                }
            }

            /* Запись лучшего результата */
            size_t trad_total = header_size + compressed_size;
            if (best_total > 0 && best_total < trad_total && best_total < input_size) {
                if (best_is_bwt && bwt_lzcm_data) {
                    /* BWT победил — записываем с форматным байтом */
                    uint16_t bwt_magic = KOLIBRI_BWT_MAGIC;
                    memcpy(out_buf, &bwt_magic, 2);
                    out_buf[2] = (uint8_t)(input_size & 0xFF);
                    out_buf[3] = (uint8_t)((input_size >> 8) & 0xFF);
                    out_buf[4] = (uint8_t)((input_size >> 16) & 0xFF);
                    uint32_t pidx32 = (uint32_t)bwt_pidx;
                    memcpy(out_buf + 5, &pidx32, 4);
                    out_buf[9] = (uint8_t)bwt_fmt;

                    if (bwt_fmt == KOLIBRI_BWT_FMT_LZCM) {
                        /* LZCM: пропускаем 0x00 single-block prefix */
                        memcpy(out_buf + KOLIBRI_BWT_HDR_SIZE,
                               bwt_lzcm_data + 1, bwt_lzcm_data_size - 1);
                    } else {
                        /* Pure CM: копируем как есть */
                        memcpy(out_buf + KOLIBRI_BWT_HDR_SIZE,
                               bwt_lzcm_data, bwt_lzcm_data_size);
                    }

                    free(bwt_lzcm_data);
                    free(lzcm_buf);
                    free(lz_buf);
                    free(temp);
                    free(token_buf);
                    token_dict_free(&tdict);

                    *output = out_buf;
                    *output_size = best_total;

                    if (stats) {
                        stats->original_size = input_size;
                        stats->compressed_size = best_total;
                        stats->compression_ratio = (double)input_size / (double)best_total;
                        stats->checksum = kolibri_checksum(input, input_size);
                        stats->file_type = file_type;
                        stats->methods_used = KOLIBRI_COMPRESS_LZCM | KOLIBRI_COMPRESS_BWT;
                        stats->compression_time_ms = get_time_ms() - start_time;
                        stats->decompression_time_ms = 0;
                    }
                    return 0;
                } else {
                    /* Обычный LZCM победил */
                    free(bwt_lzcm_data);
                    uint16_t lzcm_magic = KOLIBRI_LZCM_MAGIC;
                    memcpy(out_buf, &lzcm_magic, 2);
                    out_buf[2] = (uint8_t)(input_size & 0xFF);
                    out_buf[3] = (uint8_t)((input_size >> 8) & 0xFF);
                    out_buf[4] = (uint8_t)((input_size >> 16) & 0xFF);
                    memcpy(out_buf + KOLIBRI_LZCM_HDR_SIZE, lzcm_buf + 1, lzcm_size - 1);

                    free(lzcm_buf);
                    free(lz_buf);
                    free(temp);
                    free(token_buf);
                    token_dict_free(&tdict);

                    *output = out_buf;
                    *output_size = best_total;

                    if (stats) {
                        stats->original_size = input_size;
                        stats->compressed_size = best_total;
                        stats->compression_ratio = (double)input_size / (double)best_total;
                        stats->checksum = kolibri_checksum(input, input_size);
                        stats->file_type = file_type;
                        stats->methods_used = KOLIBRI_COMPRESS_LZCM;
                        stats->compression_time_ms = get_time_ms() - start_time;
                        stats->decompression_time_ms = 0;
                    }
                    return 0;
                }
            }
            free(bwt_lzcm_data);
            /* Fallback: LZCM с традиционным заголовком */
            if (lzcm_size > 0 && lzcm_size < compressed_size) {
                methods_used = KOLIBRI_COMPRESS_LZCM;
                memcpy(compressed_data, lzcm_buf, lzcm_size);
                compressed_size = lzcm_size;
            }
            free(lzcm_buf);
        }
    }


compress_done:
    /* v81: cleanup спекулятивного LZCM потока */
#if KF_USE_THREADS
    if (lzcm_bg_started) {
        pthread_join(lzcm_bg_tid, NULL);
        lzcm_bg_started = 0;
    }
#endif
    free(lzcm_buf_bg);
    free(lz_buf);
    free(temp);
    free(token_buf);
    token_dict_free(&tdict);

    /* Fill header (v66: compact 16 bytes) */
    KolibriCompressHeader *header = (KolibriCompressHeader *)out_buf;
    header->magic = KOLIBRI_COMPRESS_MAGIC;
    header->version = (uint16_t)KOLIBRI_COMPRESS_VERSION;
    header->methods = (uint16_t)methods_used;
    header->original_size = (uint32_t)input_size;
    header->checksum = kolibri_checksum(input, input_size);

    *output = out_buf;
    *output_size = header_size + compressed_size;

    /* Fill statistics */
    if (stats) {
        stats->original_size = input_size;
        stats->compressed_size = *output_size;
        stats->compression_ratio = (double)input_size / (double)*output_size;
        stats->checksum = header->checksum;
        stats->file_type = file_type;
        stats->methods_used = methods_used;
        stats->compression_time_ms = get_time_ms() - start_time;
        stats->decompression_time_ms = 0;
    }

    return 0;
}

int kolibri_decompress(const uint8_t *input,
                       size_t input_size,
                       uint8_t **output,
                       size_t *output_size,
                       KolibriCompressStats *stats) {
    if (!input || !output || !output_size || input_size < KOLIBRI_LZCM_HDR_SIZE) {
        return -1;
    }

    double start_time = get_time_ms();

    /* === Минимальный LZCM формат (6-байтный заголовок) === */
    uint16_t first_magic;
    memcpy(&first_magic, input, 2);
    if (first_magic == KOLIBRI_LZCM_MAGIC) {
        /* v67: 3-байтный размер (до 16MB) */
        uint32_t original_size = (uint32_t)input[2]
                               | ((uint32_t)input[3] << 8)
                               | ((uint32_t)input[4] << 16);
        if (original_size == 0 || original_size > 16 * 1024 * 1024) return -1;

        const uint8_t *lzcm_data = input + KOLIBRI_LZCM_HDR_SIZE;
        size_t lzcm_data_size = input_size - KOLIBRI_LZCM_HDR_SIZE;

        uint8_t *out_buf = (uint8_t *)malloc(original_size);
        if (!out_buf) return -1;

        size_t dec_size = decompress_lzcm_v66_block(
            lzcm_data, lzcm_data_size, out_buf, original_size, original_size);
        if (dec_size == 0 || dec_size != original_size) {
            free(out_buf);
            return -1;
        }

        *output = out_buf;
        *output_size = dec_size;

        if (stats) {
            stats->original_size = original_size;
            stats->compressed_size = input_size;
            stats->compression_ratio = (double)original_size / (double)input_size;
            stats->checksum = kolibri_checksum(out_buf, dec_size);
            stats->file_type = KOLIBRI_FILE_TEXT;
            stats->methods_used = KOLIBRI_COMPRESS_LZCM;
            stats->compression_time_ms = 0;
            stats->decompression_time_ms = get_time_ms() - start_time;
        }
        return 0;
    }

    /* === v70: BWT формат (10-байтный заголовок: KB + size(3) + pidx(4) + fmt(1)) === */
    #define KOLIBRI_BWT_HDR_SIZE_D 10
    #define KOLIBRI_BWT_FMT_LZCM_D 0
    #define KOLIBRI_BWT_FMT_CM_D   1
    #define KOLIBRI_BWT_FMT_MTF_D  2
    #define KOLIBRI_BWT_FMT_DELTA_D 3  /* v72: Delta + CM */
    #define KOLIBRI_BWT_FMT_MTF_LZCM_D 4  /* v72: MTF + LZCM */
    #define KOLIBRI_BWT_FMT_MTF_RLE_D  5  /* v77: MTF + Zero-RLE + CM */
    #define KOLIBRI_BWT_FMT_MTF_RLE_MB_D 6  /* v78: Multi-block BWT + MTF + RLE + CM */
    #define KOLIBRI_BWT_FMT_MTF_RLE_TURBO_D 7  /* v85: MTF + Zero-RLE + Turbo RC */
    if (first_magic == KOLIBRI_BWT_MAGIC && input_size >= KOLIBRI_BWT_HDR_SIZE_D) {
        uint32_t original_size = (uint32_t)input[2]
                               | ((uint32_t)input[3] << 8)
                               | ((uint32_t)input[4] << 16);
        if (original_size == 0 || original_size > 16 * 1024 * 1024) return -1;

        uint32_t pidx;
        memcpy(&pidx, input + 5, 4);
        uint8_t bwt_format = input[9];

        const uint8_t *payload = input + KOLIBRI_BWT_HDR_SIZE_D;
        size_t payload_size = input_size - KOLIBRI_BWT_HDR_SIZE_D;

        /* Шаг 1: Декомпрессируем → MTF/BWT данные */
        uint8_t *bwt_buf = (uint8_t *)malloc(original_size);
        if (!bwt_buf) return -1;

        size_t dec_size;
        if (bwt_format == KOLIBRI_BWT_FMT_CM_D
            || bwt_format == KOLIBRI_BWT_FMT_MTF_D
            || bwt_format == KOLIBRI_BWT_FMT_DELTA_D) {
            /* v72: BWT + CM (fmt=1), BWT + MTF + CM (fmt=2), BWT + Delta + CM (fmt=3) */
            dec_size = decompress_formula_v62_block(
                payload, payload_size, bwt_buf, original_size, original_size);
        } else if (bwt_format == KOLIBRI_BWT_FMT_MTF_RLE_D) {
            /* v77: BWT + MTF + Zero-RLE + CM (fmt=5) */
            /* payload содержит: rle_size(3) + CM-data */
            if (payload_size < 3) { free(bwt_buf); return -1; }
            uint32_t rle_size = (uint32_t)payload[0]
                              | ((uint32_t)payload[1] << 8)
                              | ((uint32_t)payload[2] << 16);
            if (rle_size == 0 || rle_size > original_size * 2) {
                free(bwt_buf); return -1;
            }
            /* Декомпрессируем CM → RLE поток (v82: fast path без match model) */
            uint8_t *rle_buf = (uint8_t *)malloc(rle_size);
            if (!rle_buf) { free(bwt_buf); return -1; }
            size_t rle_dec = decompress_formula_v62_block_fast(
                payload + 3, payload_size - 3, rle_buf, rle_size, rle_size);
            if (rle_dec != rle_size) { free(rle_buf); free(bwt_buf); return -1; }
            /* Раскодируем Zero-RLE → MTF поток */
            dec_size = kolibri_zero_rle_decode(
                rle_buf, rle_size, bwt_buf, original_size);
            free(rle_buf);
        } else if (bwt_format == KOLIBRI_BWT_FMT_MTF_RLE_TURBO_D) {
            /* v85: BWT + MTF + Zero-RLE + Turbo RC (fmt=7) */
            if (payload_size < 3) { free(bwt_buf); return -1; }
            uint32_t rle_size = (uint32_t)payload[0]
                              | ((uint32_t)payload[1] << 8)
                              | ((uint32_t)payload[2] << 16);
            if (rle_size == 0 || rle_size > original_size * 2) {
                free(bwt_buf); return -1;
            }
            /* Декомпрессируем Turbo RC → RLE поток */
            uint8_t *rle_buf = (uint8_t *)malloc(rle_size);
            if (!rle_buf) { free(bwt_buf); return -1; }
            size_t rle_dec = decompress_turbo_block(
                payload + 3, payload_size - 3, rle_buf, rle_size, rle_size);
            if (rle_dec != rle_size) { free(rle_buf); free(bwt_buf); return -1; }
            /* Раскодируем Zero-RLE → MTF поток */
            dec_size = kolibri_zero_rle_decode(
                rle_buf, rle_size, bwt_buf, original_size);
            free(rle_buf);
        } else if (bwt_format == KOLIBRI_BWT_FMT_MTF_RLE_MB_D) {
            /* v78: Multi-block BWT + MTF + RLE + CM (fmt=6)
             * payload: [nblocks:1] [per-block: bs(3)+pidx(4)+rle_sz(3)] [CM_data] */
            if (payload_size < 1) { free(bwt_buf); return -1; }
            size_t nblocks = payload[0];
            if (nblocks < 2 || nblocks > 255) { free(bwt_buf); return -1; }
            size_t meta_sz = 1 + nblocks * 10;
            if (payload_size < meta_sz) { free(bwt_buf); return -1; }

            /* Читаем метаданные блоков */
            size_t *mb_bs = (size_t *)malloc(nblocks * sizeof(size_t));
            saidx_t *mb_pidx = (saidx_t *)malloc(nblocks * sizeof(saidx_t));
            size_t *mb_rle_sz = (size_t *)malloc(nblocks * sizeof(size_t));
            if (!mb_bs || !mb_pidx || !mb_rle_sz) {
                free(mb_bs); free(mb_pidx); free(mb_rle_sz);
                free(bwt_buf); return -1;
            }

            size_t total_rle = 0, total_orig = 0;
            for (size_t bi = 0; bi < nblocks; bi++) {
                size_t mi = 1 + bi * 10;
                mb_bs[bi] = (size_t)payload[mi]
                          | ((size_t)payload[mi+1] << 8)
                          | ((size_t)payload[mi+2] << 16);
                uint32_t pidx32;
                memcpy(&pidx32, payload + mi + 3, 4);
                mb_pidx[bi] = (saidx_t)pidx32;
                mb_rle_sz[bi] = (size_t)payload[mi+7]
                              | ((size_t)payload[mi+8] << 8)
                              | ((size_t)payload[mi+9] << 16);
                total_rle += mb_rle_sz[bi];
                total_orig += mb_bs[bi];
            }

            if (total_orig != original_size) {
                free(mb_bs); free(mb_pidx); free(mb_rle_sz);
                free(bwt_buf); return -1;
            }

            /* CM decompress → total_rle bytes of concatenated RLE data */
            uint8_t *rle_total = (uint8_t *)malloc(total_rle);
            if (!rle_total) {
                free(mb_bs); free(mb_pidx); free(mb_rle_sz);
                free(bwt_buf); return -1;
            }
            size_t rle_dec = decompress_formula_v62_block_ex(
                payload + meta_sz, payload_size - meta_sz,
                rle_total, total_rle, total_rle, 0);
            if (rle_dec != total_rle) {
                free(rle_total); free(mb_bs); free(mb_pidx); free(mb_rle_sz);
                free(bwt_buf); return -1;
            }

            /* Per-block: RLE decode → MTF inverse → BWT inverse → output */
            size_t rle_off = 0, out_off = 0;
            int mb_fail = 0;
            for (size_t bi = 0; bi < nblocks && !mb_fail; bi++) {
                uint8_t *blk = (uint8_t *)malloc(mb_bs[bi]);
                if (!blk) { mb_fail = 1; break; }

                /* Zero-RLE decode */
                size_t mtf_sz = kolibri_zero_rle_decode(
                    rle_total + rle_off, mb_rle_sz[bi], blk, mb_bs[bi]);
                rle_off += mb_rle_sz[bi];
                if (mtf_sz != mb_bs[bi]) { free(blk); mb_fail = 1; break; }

                /* MTF inverse */
                kolibri_mtf_inverse(blk, mb_bs[bi]);

                /* BWT inverse → bwt_buf + out_off */
                uint8_t *ibwt = (uint8_t *)malloc(mb_bs[bi]);
                if (!ibwt) { free(blk); mb_fail = 1; break; }

                if (inverse_bw_transform(blk, ibwt, NULL,
                                         (saidx_t)mb_bs[bi], mb_pidx[bi]) != 0) {
                    free(ibwt); free(blk); mb_fail = 1; break;
                }

                memcpy(bwt_buf + out_off, ibwt, mb_bs[bi]);
                free(ibwt);
                free(blk);
                out_off += mb_bs[bi];
            }

            free(rle_total); free(mb_bs); free(mb_pidx); free(mb_rle_sz);

            if (mb_fail) { free(bwt_buf); return -1; }
            dec_size = original_size;
            /* bwt_buf now contains fully decoded output — skip MTF/BWT steps */
        } else if (bwt_format == KOLIBRI_BWT_FMT_MTF_LZCM_D) {
            /* v72: BWT + MTF + LZCM (fmt=4) */
            dec_size = decompress_lzcm_v66_block(
                payload, payload_size, bwt_buf, original_size, original_size);
        } else {
            /* BWT + LZCM (fmt=0, legacy) */
            dec_size = decompress_lzcm_v66_block(
                payload, payload_size, bwt_buf, original_size, original_size);
        }
        if (dec_size == 0 || dec_size != original_size) {
            free(bwt_buf);
            return -1;
        }

        /* Шаг 1.5: Обратный MTF/Delta (пропускаем для fmt=6, уже обработано per-block) */
        if (bwt_format == KOLIBRI_BWT_FMT_MTF_RLE_MB_D) {
            /* fmt=6: всё уже декодировано в decompression handler */
        } else if (bwt_format == KOLIBRI_BWT_FMT_MTF_D
            || bwt_format == KOLIBRI_BWT_FMT_MTF_LZCM_D
            || bwt_format == KOLIBRI_BWT_FMT_MTF_RLE_D
            || bwt_format == KOLIBRI_BWT_FMT_MTF_RLE_TURBO_D) {  /* v85: +turbo */
            kolibri_mtf_inverse(bwt_buf, original_size);
        } else if (bwt_format == KOLIBRI_BWT_FMT_DELTA_D) {
            kolibri_delta_inverse(bwt_buf, original_size);
        }

        /* Шаг 2: Обратное BWT преобразование (пропускаем для fmt=6) */
        if (bwt_format == KOLIBRI_BWT_FMT_MTF_RLE_MB_D) {
            /* fmt=6: bwt_buf уже содержит финальный результат */
            *output = bwt_buf;
            *output_size = original_size;
        } else {
            uint8_t *out_buf = (uint8_t *)malloc(original_size);
            if (!out_buf) { free(bwt_buf); return -1; }

            if (inverse_bw_transform(bwt_buf, out_buf, NULL,
                                      (saidx_t)original_size, (saidx_t)pidx) != 0) {
                free(bwt_buf);
                free(out_buf);
                return -1;
            }
            free(bwt_buf);

            *output = out_buf;
            *output_size = original_size;
        }

        if (stats) {
            stats->original_size = original_size;
            stats->compressed_size = input_size;
            stats->compression_ratio = (double)original_size / (double)input_size;
            stats->checksum = kolibri_checksum(*output, original_size);
            stats->file_type = KOLIBRI_FILE_TEXT;
            stats->methods_used = KOLIBRI_COMPRESS_LZCM | KOLIBRI_COMPRESS_BWT;
            stats->compression_time_ms = 0;
            stats->decompression_time_ms = get_time_ms() - start_time;
        }
        return 0;
    }

    /* === Традиционный формат (16-байтный заголовок) === */
    if (input_size < sizeof(KolibriCompressHeader)) {
        return -1;
    }

    /* Read and verify header */
    const KolibriCompressHeader *header = (const KolibriCompressHeader *)input;
    if (header->magic != KOLIBRI_COMPRESS_MAGIC) {
        return -1; /* Invalid format */
    }
    /* Support versions 1-66 for backward compatibility */
    if (header->version < 1 || header->version > KOLIBRI_COMPRESS_VERSION) {
        return -1; /* Unsupported version */
    }

    const uint8_t *compressed_data = input + sizeof(KolibriCompressHeader);
    size_t compressed_size = input_size - sizeof(KolibriCompressHeader);
    size_t original_size = header->original_size;

    /* === Пустой файл: original_size == 0, только заголовок === */
    if (original_size == 0) {
        /* Проверяем контрольную сумму пустых данных */
        uint32_t empty_checksum = kolibri_checksum(NULL, 0);
        if (header->checksum != empty_checksum) {
            return -1;
        }
        uint8_t *out_buf = (uint8_t *)malloc(1); /* non-NULL sentinel */
        if (!out_buf) return -1;
        *output = out_buf;
        *output_size = 0;
        if (stats) {
            stats->original_size = 0;
            stats->compressed_size = input_size;
            stats->compression_ratio = 1.0;
            stats->checksum = empty_checksum;
            stats->file_type = KOLIBRI_FILE_UNKNOWN;
            stats->methods_used = 0;
            stats->compression_time_ms = 0;
            stats->decompression_time_ms = get_time_ms() - start_time;
        }
        return 0;
    }

    /* Allocate output buffer */
    uint8_t *out_buf = (uint8_t *)malloc(original_size);
    if (!out_buf) return -1;

    /* Allocate temporary buffers */
    uint8_t *temp1 = (uint8_t *)malloc(original_size * 2);
    uint8_t *temp2 = (uint8_t *)malloc(original_size * 2);
    if (!temp1 || !temp2) {
        free(out_buf);
        free(temp1);
        free(temp2);
        return -1;
    }

    /* Copy compressed data to temp buffer */
    memcpy(temp1, compressed_data, compressed_size);
    size_t current_size = compressed_size;
    const uint8_t *current_data = temp1;

    /* Token dictionary (v52) */
    TokenDict tdict = {0};
    size_t token_size = original_size;
    int token_used = 0;
    if ((header->methods & KOLIBRI_COMPRESS_TOKEN) && header->version >= 52) {
        size_t dict_consumed = 0;
        if (!token_dict_read(current_data, current_size, &dict_consumed, &tdict)) {
            free(out_buf); free(temp1); free(temp2);
            return -1;
        }
        current_data += dict_consumed;
        current_size -= dict_consumed;
        if (current_size < 4) {
            token_dict_free(&tdict);
            free(out_buf); free(temp1); free(temp2);
            return -1;
        }
        uint32_t token_intermediate = 0;
        memcpy(&token_intermediate, current_data, 4);
        token_size = (size_t)token_intermediate;
        current_data += 4;
        current_size -= 4;
        token_used = 1;
    }

    /* Decompress in reverse order of compression */

    /* === v51: Формульная декомпрессия === */
    if (header->methods & KOLIBRI_COMPRESS_FORMULA) {
        /* Если и LZ и Formula — первые 4 байта = размер LZ-промежуточных данных */
        size_t formula_target = token_used ? token_size : original_size;
        const uint8_t *formula_data = current_data;
        size_t formula_data_size = current_size;

        if ((header->methods & KOLIBRI_COMPRESS_LZ77) && header->version >= 51) {
            /* Читаем lz_intermediate_size из префикса */
            if (current_size < 4) {
                token_dict_free(&tdict);
                free(out_buf); free(temp1); free(temp2);
                return -1;
            }
            uint32_t lz_intermediate;
            memcpy(&lz_intermediate, current_data, 4);
            formula_target = (size_t)lz_intermediate;
            formula_data = current_data + 4;
            formula_data_size = current_size - 4;
        }

        size_t formula_size;
        if (header->version >= 62) {
            formula_size = decompress_formula_v62(
                formula_data, formula_data_size, temp2, original_size * 2, formula_target);
        } else {
            formula_size = decompress_formula_v51(
                formula_data, formula_data_size, temp2, original_size * 2, formula_target);
        }
        if (formula_size == 0 || formula_size != formula_target) {
            token_dict_free(&tdict);
            free(out_buf); free(temp1); free(temp2);
            return -1;
        }
        memcpy(temp1, temp2, formula_size);
        current_size = formula_size;
        current_data = temp1;
    }

    /* === v66: LZCM unified decoder === */
    if (header->methods & KOLIBRI_COMPRESS_LZCM) {
        size_t lzcm_target = original_size;
        size_t lzcm_size = decompress_lzcm_v66(
            current_data, current_size, temp2, original_size * 2, lzcm_target);
        if (lzcm_size == 0 || lzcm_size != lzcm_target) {
            token_dict_free(&tdict);
            free(out_buf); free(temp1); free(temp2);
            return -1;
        }
        memcpy(temp1, temp2, lzcm_size);
        current_size = lzcm_size;
        current_data = temp1;
    }

    /* Huffman/ANS decompression (backward compat) */
    if (header->methods & KOLIBRI_COMPRESS_HUFFMAN) {
        size_t huff_size = kolibri_entropy_decompress(current_data, current_size, temp2, original_size * 2);
        if (huff_size == 0) {
            token_dict_free(&tdict);
            free(out_buf);
            free(temp1);
            free(temp2);
            return -1;
        }
        memcpy(temp1, temp2, huff_size);
        current_size = huff_size;
        current_data = temp1;
    }

    /* RLE decompression */
    if (header->methods & KOLIBRI_COMPRESS_RLE) {
        size_t rle_size = decompress_rle(current_data, current_size, temp2, original_size * 2);
        if (rle_size == 0) {
            token_dict_free(&tdict);
            free(out_buf);
            free(temp1);
            free(temp2);
            return -1;
        }
        memcpy(temp1, temp2, rle_size);
        current_size = rle_size;
        current_data = temp1;
    }

    /* LZ-lite decompression (v51 hybrid) или старый LZ77 (backward compat) */
    if (header->methods & KOLIBRI_COMPRESS_LZ77) {
        size_t lz_size;
        if (header->version >= 51) {
            /* v51+: LZ-lite формат */
            size_t lz_target = token_used ? token_size : original_size;
            lz_size = lz_lite_decode(current_data, current_size,
                                     temp2, original_size * 2, lz_target);
        } else {
            /* v50 и ранее: старый LZ77 */
            lz_size = decompress_lz77(current_data, current_size, temp2, original_size * 2);
        }
        if (lz_size == 0) {
            token_dict_free(&tdict);
            free(out_buf);
            free(temp1);
            free(temp2);
            return -1;
        }
        memcpy(temp1, temp2, lz_size);
        current_size = lz_size;
        current_data = temp1;
    }

    /* Token detokenization (v52) */
    if (token_used) {
        size_t detok_size = token_decode_text(current_data, current_size,
                                              temp2, original_size * 2,
                                              &tdict, original_size);
        if (detok_size == 0 || detok_size != original_size) {
            token_dict_free(&tdict);
            free(out_buf); free(temp1); free(temp2);
            return -1;
        }
        memcpy(temp1, temp2, detok_size);
        current_size = detok_size;
        current_data = temp1;
    }

    /* Mathematical decompression */
    if (header->methods & KOLIBRI_COMPRESS_MATH) {
        /* Pass strict original_size as limit to ensure block logic works correctly */
        size_t math_size = decompress_mathematical(current_data, current_size, temp2, original_size, header->version);
        
        if (math_size == 0 || math_size != original_size) {
            token_dict_free(&tdict);
            free(out_buf);
            free(temp1);
            free(temp2);
            return -1;
        }
        memcpy(temp1, temp2, math_size);
        current_size = math_size;
        current_data = temp1;
    }

    /* Copy final decompressed data */
    memcpy(out_buf, current_data, current_size);

    /* Verify checksum (skip if 0 — turbo mode) */
    if (header->checksum != 0) {
        uint32_t checksum = kolibri_checksum(out_buf, current_size);
        if (checksum != header->checksum) {
            token_dict_free(&tdict);
            free(out_buf);
            free(temp1);
            free(temp2);
            return -1; /* Checksum mismatch */
        }
    }

    *output = out_buf;
    *output_size = current_size;

    token_dict_free(&tdict);
    free(temp1);
    free(temp2);

    /* Fill statistics */
    if (stats) {
        stats->original_size = original_size;
        stats->compressed_size = input_size;
        stats->compression_ratio = (double)original_size / (double)input_size;
        stats->checksum = header->checksum;
        stats->file_type = KOLIBRI_FILE_TEXT;
        stats->methods_used = header->methods;
        stats->compression_time_ms = 0;
        stats->decompression_time_ms = get_time_ms() - start_time;
    }

    return 0;
}

/* Archive management implementation */
#define KOLIBRI_ARCHIVE_MAGIC 0x4B415243 /* "KARC" */
#define KOLIBRI_ARCHIVE_VERSION 50
#define KOLIBRI_ARCHIVE_MAX_ENTRIES 1024

typedef struct {
    KolibriArchiveEntry entry;
    size_t data_offset;
    size_t data_size;
} KolibriArchiveEntryInternal;

struct KolibriArchive {
    char filename[512];
    FILE *file;
    KolibriArchiveEntryInternal entries[KOLIBRI_ARCHIVE_MAX_ENTRIES];
    size_t entry_count;
    int mode; /* 0 = read, 1 = write */
};

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t entry_count;
    uint8_t reserved[52];
} KolibriArchiveHeader;

KolibriArchive *kolibri_archive_create(const char *filename) {
    if (!filename) return NULL;

    KolibriArchive *archive = (KolibriArchive *)calloc(1, sizeof(KolibriArchive));
    if (!archive) return NULL;

    strncpy(archive->filename, filename, sizeof(archive->filename) - 1);
    archive->file = fopen(filename, "wb");
    if (!archive->file) {
        free(archive);
        return NULL;
    }

    archive->mode = 1; /* write mode */
    archive->entry_count = 0;

    /* Write placeholder header */
    KolibriArchiveHeader header = {0};
    header.magic = KOLIBRI_ARCHIVE_MAGIC;
    header.version = KOLIBRI_ARCHIVE_VERSION;
    fwrite(&header, sizeof(header), 1, archive->file);

    /* Reserve space for entries table following the header */
    uint8_t *placeholder = (uint8_t *)calloc(KOLIBRI_ARCHIVE_MAX_ENTRIES, sizeof(KolibriArchiveEntryInternal));
    if (placeholder) {
        fwrite(placeholder, sizeof(KolibriArchiveEntryInternal), KOLIBRI_ARCHIVE_MAX_ENTRIES, archive->file);
        free(placeholder);
    }

    return archive;
}

KolibriArchive *kolibri_archive_open(const char *filename) {
    if (!filename) return NULL;

    FILE *file = fopen(filename, "rb");
    if (!file) return NULL;

    KolibriArchiveHeader header;
    if (fread(&header, sizeof(header), 1, file) != 1 ||
        header.magic != KOLIBRI_ARCHIVE_MAGIC ||
        header.version < 1 || header.version > KOLIBRI_ARCHIVE_VERSION) {
        fclose(file);
        return NULL;
    }

    KolibriArchive *archive = (KolibriArchive *)calloc(1, sizeof(KolibriArchive));
    if (!archive) {
        fclose(file);
        return NULL;
    }

    strncpy(archive->filename, filename, sizeof(archive->filename) - 1);
    archive->file = file;
    archive->mode = 0; /* read mode */
    archive->entry_count = header.entry_count;

    /* Read entry table */
    for (size_t i = 0; i < header.entry_count && i < KOLIBRI_ARCHIVE_MAX_ENTRIES; i++) {
        if (fread(&archive->entries[i], sizeof(KolibriArchiveEntryInternal), 1, file) != 1) {
            fclose(file);
            free(archive);
            return NULL;
        }
    }

    return archive;
}

int kolibri_archive_add_file(KolibriArchive *archive,
                              const char *filename,
                              const uint8_t *data,
                              size_t size) {
    if (!archive || !filename || !data || archive->mode != 1) {
        return -1;
    }

    if (archive->entry_count >= KOLIBRI_ARCHIVE_MAX_ENTRIES) {
        return -1; /* Archive full */
    }

    /* Compress the data */
    KolibriCompressor *comp = kolibri_compressor_create(KOLIBRI_COMPRESS_ALL);
    if (!comp) return -1;

    uint8_t *compressed = NULL;
    size_t compressed_size = 0;
    KolibriCompressStats stats;

    int ret = kolibri_compress(comp, data, size, &compressed, &compressed_size, &stats);
    kolibri_compressor_destroy(comp);

    if (ret != 0) {
        return -1;
    }

    /* Get current file position for data offset */
    long data_offset = ftell(archive->file);
    if (data_offset < 0) {
        free(compressed);
        return -1;
    }

    /* Write compressed data */
    if (fwrite(compressed, 1, compressed_size, archive->file) != compressed_size) {
        free(compressed);
        return -1;
    }

    /* Add entry */
    KolibriArchiveEntryInternal *entry = &archive->entries[archive->entry_count];
    strncpy(entry->entry.name, filename, sizeof(entry->entry.name) - 1);
    entry->entry.original_size = size;
    entry->entry.compressed_size = compressed_size;
    entry->entry.checksum = stats.checksum;
    entry->entry.timestamp = (uint64_t)time(NULL);
    entry->entry.type = stats.file_type;
    entry->data_offset = (size_t)data_offset;
    entry->data_size = compressed_size;

    archive->entry_count++;

    free(compressed);
    return 0;
}

int kolibri_archive_extract_file(KolibriArchive *archive,
                                  const char *filename,
                                  uint8_t **data,
                                  size_t *size) {
    if (!archive || !filename || !data || !size) {
        return -1;
    }

    /* Find entry */
    KolibriArchiveEntryInternal *entry = NULL;
    for (size_t i = 0; i < archive->entry_count; i++) {
        if (strcmp(archive->entries[i].entry.name, filename) == 0) {
            entry = &archive->entries[i];
            break;
        }
    }

    if (!entry) {
        return -1; /* File not found */
    }

    /* Seek to data */
    if (fseek(archive->file, (long)entry->data_offset, SEEK_SET) != 0) {
        return -1;
    }

    /* Защита: проверка размера перед аллокацией (макс. 512МБ) */
    if (entry->data_size == 0 || entry->data_size > (512UL * 1024UL * 1024UL)) {
        return -1;
    }

    /* Read compressed data */
    uint8_t *compressed = (uint8_t *)malloc(entry->data_size);
    if (!compressed) return -1;

    if (fread(compressed, 1, entry->data_size, archive->file) != entry->data_size) {
        free(compressed);
        return -1;
    }

    /* Decompress */
    uint8_t *decompressed = NULL;
    size_t decompressed_size = 0;
    int ret = kolibri_decompress(compressed, entry->data_size, &decompressed, 
                                  &decompressed_size, NULL);
    free(compressed);

    if (ret != 0) {
        return -1;
    }

    *data = decompressed;
    *size = decompressed_size;

    return 0;
}

int kolibri_archive_list(KolibriArchive *archive,
                         KolibriArchiveEntry **entries,
                         size_t *count) {
    if (!archive || !entries || !count) {
        return -1;
    }

    if (archive->entry_count == 0) {
        *entries = NULL;
        *count = 0;
        return 0;
    }

    /* Allocate and copy entries */
    KolibriArchiveEntry *entry_list = (KolibriArchiveEntry *)malloc(
        archive->entry_count * sizeof(KolibriArchiveEntry));
    if (!entry_list) return -1;

    for (size_t i = 0; i < archive->entry_count; i++) {
        entry_list[i] = archive->entries[i].entry;
    }

    *entries = entry_list;
    *count = archive->entry_count;

    return 0;
}

void kolibri_archive_close(KolibriArchive *archive) {
    if (!archive) return;

    if (archive->mode == 1) {
        /* Write mode - update header and entry table */
        fseek(archive->file, 0, SEEK_SET);

        KolibriArchiveHeader header = {0};
        header.magic = KOLIBRI_ARCHIVE_MAGIC;
        header.version = KOLIBRI_ARCHIVE_VERSION;
        header.entry_count = (uint32_t)archive->entry_count;
        fwrite(&header, sizeof(header), 1, archive->file);

        /* Write entry table */
        for (size_t i = 0; i < archive->entry_count; i++) {
            fwrite(&archive->entries[i], sizeof(KolibriArchiveEntryInternal), 1, archive->file);
        }
    }

    if (archive->file) {
        fclose(archive->file);
    }

    free(archive);
}

/* ============================================================================
 * Streaming (потоковое) API — инкрементальное сжатие/распаковка
 * ============================================================================ */

/* --- Магический маркер потокового формата --- */
#define KOLIBRI_STREAM_MAGIC 0x4B53  /* "KS" */
#define KOLIBRI_STREAM_VERSION 1
/* Заголовок потока: magic(2) + version(1) + block_size(4) = 7 байт */
#define KOLIBRI_STREAM_HDR_SIZE 7
/* Маркер конца потока: block_csize == 0xFFFFFFFF */
#define KOLIBRI_STREAM_END_MARKER 0xFFFFFFFFu

struct KolibriStream {
    KolibriStreamMode mode;
    uint32_t methods;
    KolibriStreamWriteFn write_fn;
    void *user_data;

    /* Аккумулятор входных данных (до block_size) */
    uint8_t *buf;
    size_t buf_pos;
    size_t block_size;

    /* Статистика */
    size_t total_in;
    size_t total_out;
    int blocks_written;
    int finished;

    /* Для декомпрессии: чтение заголовка + блоков */
    int header_read;
    /* Буфер заголовка блока (4 байта compressed_size + 4 байта original_size) */
    uint8_t blk_hdr[8];
    size_t blk_hdr_pos;
    uint8_t *blk_buf;
    size_t blk_buf_size;
    size_t blk_buf_pos;
    size_t blk_orig_size;
    int reading_blk_hdr;

    /* Буфер для чтения заголовка потока (7 байт) */
    uint8_t stream_hdr[KOLIBRI_STREAM_HDR_SIZE];
    size_t stream_hdr_pos;
};

KolibriStream *kolibri_stream_create(KolibriStreamMode mode,
                                      uint32_t methods,
                                      KolibriStreamWriteFn write_fn,
                                      void *user_data) {
    if (!write_fn) return NULL;

    KolibriStream *s = (KolibriStream *)calloc(1, sizeof(KolibriStream));
    if (!s) return NULL;

    s->mode = mode;
    s->methods = methods;
    s->write_fn = write_fn;
    s->user_data = user_data;
    s->block_size = (size_t)KF62_BLOCK_SIZE;
    s->finished = 0;

    s->buf = (uint8_t *)malloc(s->block_size);
    if (!s->buf) { free(s); return NULL; }
    s->buf_pos = 0;

    if (mode == KOLIBRI_STREAM_DECOMPRESS) {
        s->reading_blk_hdr = 1;
        s->blk_hdr_pos = 0;
        s->header_read = 0;
        s->stream_hdr_pos = 0;
    }

    return s;
}

/* --- Вспомогательная: сжать один блок и отправить через write_fn --- */
static KolibriStreamStatus stream_flush_compress_block(KolibriStream *s,
                                                         const uint8_t *data,
                                                         size_t data_size) {
    if (data_size == 0) return KOLIBRI_STREAM_OK;

    /* Сжимаем блок через LZCM */
    size_t out_max = data_size + 1024;
    uint8_t *cbuf = (uint8_t *)malloc(out_max);
    if (!cbuf) return KOLIBRI_STREAM_ERROR;

    size_t csize = compress_lzcm_v66_block(data, data_size, cbuf, out_max);

    /* Заголовок блока: compressed_size(4) + original_size(4) = 8 байт */
    uint8_t blk_hdr[8];
    uint32_t cs, os;

    if (csize == 0 || csize >= data_size) {
        /* Несжимаемый: записываем raw */
        os = (uint32_t)data_size;
        cs = (uint32_t)data_size;
        memcpy(blk_hdr, &cs, 4);
        memcpy(blk_hdr + 4, &os, 4);

        int r = s->write_fn(s->user_data, blk_hdr, 8);
        if (r != 0) { free(cbuf); return KOLIBRI_STREAM_ERROR; }
        s->total_out += 8;

        r = s->write_fn(s->user_data, data, data_size);
        if (r != 0) { free(cbuf); return KOLIBRI_STREAM_ERROR; }
        s->total_out += data_size;
    } else {
        os = (uint32_t)data_size;
        cs = (uint32_t)csize;
        memcpy(blk_hdr, &cs, 4);
        memcpy(blk_hdr + 4, &os, 4);

        int r = s->write_fn(s->user_data, blk_hdr, 8);
        if (r != 0) { free(cbuf); return KOLIBRI_STREAM_ERROR; }
        s->total_out += 8;

        r = s->write_fn(s->user_data, cbuf, csize);
        if (r != 0) { free(cbuf); return KOLIBRI_STREAM_ERROR; }
        s->total_out += csize;
    }

    free(cbuf);
    s->blocks_written++;
    s->total_in += data_size;
    return KOLIBRI_STREAM_OK;
}

/* --- Вспомогательная: записать stream-заголовок --- */
static KolibriStreamStatus stream_write_header(KolibriStream *s) {
    uint8_t hdr[KOLIBRI_STREAM_HDR_SIZE];
    uint16_t magic = KOLIBRI_STREAM_MAGIC;
    memcpy(hdr, &magic, 2);
    hdr[2] = (uint8_t)KOLIBRI_STREAM_VERSION;
    uint32_t bs = (uint32_t)s->block_size;
    memcpy(hdr + 3, &bs, 4);

    int r = s->write_fn(s->user_data, hdr, KOLIBRI_STREAM_HDR_SIZE);
    if (r != 0) return KOLIBRI_STREAM_ERROR;
    s->total_out += KOLIBRI_STREAM_HDR_SIZE;
    return KOLIBRI_STREAM_OK;
}

KolibriStreamStatus kolibri_stream_write(KolibriStream *stream,
                                          const uint8_t *data,
                                          size_t size) {
    if (!stream || !data || stream->finished) return KOLIBRI_STREAM_ERROR;

    if (stream->mode == KOLIBRI_STREAM_COMPRESS) {
        /* Первый вызов — пишем заголовок потока */
        if (stream->blocks_written == 0 && stream->buf_pos == 0 && stream->total_out == 0) {
            KolibriStreamStatus st = stream_write_header(stream);
            if (st != KOLIBRI_STREAM_OK) return st;
        }

        size_t off = 0;
        while (off < size) {
            size_t space = stream->block_size - stream->buf_pos;
            size_t take = MIN(space, size - off);
            memcpy(stream->buf + stream->buf_pos, data + off, take);
            stream->buf_pos += take;
            off += take;

            /* Блок полон — сжимаем и отправляем */
            if (stream->buf_pos >= stream->block_size) {
                KolibriStreamStatus st = stream_flush_compress_block(
                    stream, stream->buf, stream->buf_pos);
                if (st != KOLIBRI_STREAM_OK) return st;
                stream->buf_pos = 0;
            }
        }
        return KOLIBRI_STREAM_OK;

    } else {
        /* === DECOMPRESS === */
        size_t off = 0;

        /* Читаем заголовок потока */
        if (!stream->header_read) {
            while (off < size && stream->stream_hdr_pos < KOLIBRI_STREAM_HDR_SIZE) {
                stream->stream_hdr[stream->stream_hdr_pos++] = data[off++];
            }
            if (stream->stream_hdr_pos < KOLIBRI_STREAM_HDR_SIZE) {
                return KOLIBRI_STREAM_NEED_MORE;
            }
            uint16_t magic;
            memcpy(&magic, stream->stream_hdr, 2);
            if (magic != KOLIBRI_STREAM_MAGIC) return KOLIBRI_STREAM_ERROR;
            uint32_t bs;
            memcpy(&bs, stream->stream_hdr + 3, 4);
            stream->block_size = (size_t)bs;
            stream->header_read = 1;
            stream->reading_blk_hdr = 1;
            stream->blk_hdr_pos = 0;
        }

        while (off < size) {
            /* Читаем заголовок блока (8 байт) */
            if (stream->reading_blk_hdr) {
                while (off < size && stream->blk_hdr_pos < 8) {
                    stream->blk_hdr[stream->blk_hdr_pos++] = data[off++];
                }
                if (stream->blk_hdr_pos < 8) return KOLIBRI_STREAM_NEED_MORE;

                uint32_t cs, os;
                memcpy(&cs, stream->blk_hdr, 4);
                memcpy(&os, stream->blk_hdr + 4, 4);

                /* Маркер конца потока */
                if (cs == KOLIBRI_STREAM_END_MARKER) {
                    stream->finished = 1;
                    return KOLIBRI_STREAM_DONE;
                }

                stream->blk_buf_size = (size_t)cs;
                stream->blk_orig_size = (size_t)os;
                stream->blk_buf = (uint8_t *)malloc(stream->blk_buf_size);
                if (!stream->blk_buf) return KOLIBRI_STREAM_ERROR;
                stream->blk_buf_pos = 0;
                stream->reading_blk_hdr = 0;
            }

            /* Читаем данные блока */
            {
                size_t need = stream->blk_buf_size - stream->blk_buf_pos;
                size_t avail = size - off;
                size_t take = MIN(need, avail);
                memcpy(stream->blk_buf + stream->blk_buf_pos, data + off, take);
                stream->blk_buf_pos += take;
                off += take;
            }

            if (stream->blk_buf_pos >= stream->blk_buf_size) {
                /* Блок полностью получен — декомпрессия */
                stream->total_in += stream->blk_buf_size;

                uint8_t *out_buf = (uint8_t *)malloc(stream->blk_orig_size);
                if (!out_buf) {
                    free(stream->blk_buf);
                    stream->blk_buf = NULL;
                    return KOLIBRI_STREAM_ERROR;
                }

                if (stream->blk_buf_size == stream->blk_orig_size) {
                    /* Raw блок — копируем */
                    memcpy(out_buf, stream->blk_buf, stream->blk_orig_size);
                } else {
                    /* Сжатый блок — распаковка */
                    size_t dec = decompress_lzcm_v66_block(
                        stream->blk_buf, stream->blk_buf_size,
                        out_buf, stream->blk_orig_size, stream->blk_orig_size);
                    if (dec != stream->blk_orig_size) {
                        free(out_buf);
                        free(stream->blk_buf);
                        stream->blk_buf = NULL;
                        return KOLIBRI_STREAM_ERROR;
                    }
                }

                /* Отдаём данные пользователю */
                int r = stream->write_fn(stream->user_data, out_buf, stream->blk_orig_size);
                stream->total_out += stream->blk_orig_size;
                stream->blocks_written++;

                free(out_buf);
                free(stream->blk_buf);
                stream->blk_buf = NULL;

                if (r != 0) return KOLIBRI_STREAM_ERROR;

                /* Готовы к следующему блоку */
                stream->reading_blk_hdr = 1;
                stream->blk_hdr_pos = 0;
            }
        }

        return KOLIBRI_STREAM_OK;
    }
}

KolibriStreamStatus kolibri_stream_finish(KolibriStream *stream) {
    if (!stream || stream->finished) return KOLIBRI_STREAM_ERROR;

    if (stream->mode == KOLIBRI_STREAM_COMPRESS) {
        /* Если ещё ничего не писали — запишем заголовок */
        if (stream->total_out == 0) {
            KolibriStreamStatus st = stream_write_header(stream);
            if (st != KOLIBRI_STREAM_OK) return st;
        }

        /* Сбрасываем оставшиеся данные */
        if (stream->buf_pos > 0) {
            KolibriStreamStatus st = stream_flush_compress_block(
                stream, stream->buf, stream->buf_pos);
            if (st != KOLIBRI_STREAM_OK) return st;
            stream->buf_pos = 0;
        }

        /* Пишем маркер конца потока: csize = 0xFFFFFFFF, osize = 0 */
        uint8_t end_marker[8];
        uint32_t end_cs = KOLIBRI_STREAM_END_MARKER;
        uint32_t end_os = 0;
        memcpy(end_marker, &end_cs, 4);
        memcpy(end_marker + 4, &end_os, 4);
        int r = stream->write_fn(stream->user_data, end_marker, 8);
        if (r != 0) return KOLIBRI_STREAM_ERROR;
        stream->total_out += 8;
    }

    stream->finished = 1;
    return KOLIBRI_STREAM_DONE;
}

void kolibri_stream_stats(const KolibriStream *stream,
                           KolibriCompressStats *stats) {
    if (!stream || !stats) return;
    memset(stats, 0, sizeof(*stats));
    if (stream->mode == KOLIBRI_STREAM_COMPRESS) {
        stats->original_size = stream->total_in;
        stats->compressed_size = stream->total_out;
    } else {
        stats->original_size = stream->total_out;
        stats->compressed_size = stream->total_in;
    }
    if (stats->compressed_size > 0) {
        stats->compression_ratio =
            (double)stats->original_size / (double)stats->compressed_size;
    }
    stats->methods_used = KOLIBRI_COMPRESS_LZCM;
}

void kolibri_stream_destroy(KolibriStream *stream) {
    if (!stream) return;
    free(stream->buf);
    free(stream->blk_buf);
    free(stream);
}
