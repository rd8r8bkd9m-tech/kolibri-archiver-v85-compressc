/*
 * Kolibri OS Archiver - Advanced Compression System
 * Implements multi-layer compression with mathematical analysis and formula-based encoding
 */

#ifndef KOLIBRI_COMPRESS_H
#define KOLIBRI_COMPRESS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version */
#define KOLIBRI_ARCHIVER_VERSION_MAJOR 76
#define KOLIBRI_ARCHIVER_VERSION_MINOR 0
#define KOLIBRI_ARCHIVER_VERSION_PATCH 0

/* Compression methods flags */
#define KOLIBRI_COMPRESS_LZ77    0x01
#define KOLIBRI_COMPRESS_RLE     0x02
#define KOLIBRI_COMPRESS_HUFFMAN 0x04
#define KOLIBRI_COMPRESS_FORMULA 0x08
#define KOLIBRI_COMPRESS_MATH    0x10
#define KOLIBRI_COMPRESS_LZMA    0x20  /* v40: LZMA compression */
#define KOLIBRI_COMPRESS_ZSTD    0x40  /* v40: Zstandard compression */
#define KOLIBRI_COMPRESS_ADAPTIVE 0x80 /* v40: Adaptive dictionary */
#define KOLIBRI_COMPRESS_TOKEN   0x100 /* v52: Token-level text stream */
#define KOLIBRI_COMPRESS_LZCM    0x200 /* v66: Unified LZ+CM (literals через CM) */
#define KOLIBRI_COMPRESS_BWT     0x400 /* v70: BWT preprocessing for text */
#define KOLIBRI_COMPRESS_TURBO   0x800  /* v75: Turbo LZ-only (max speed, no CM) */
#define KOLIBRI_COMPRESS_BLAZING 0x1000 /* v76: Blazing-fast LZ (>1 GB/s, <0.1ms/100KB) */
#define KOLIBRI_COMPRESS_ALL     0x7FF

/* File type detection */
typedef enum {
    KOLIBRI_FILE_BINARY,
    KOLIBRI_FILE_TEXT,
    KOLIBRI_FILE_IMAGE,
    KOLIBRI_FILE_UNKNOWN
} KolibriFileType;

/* Compression statistics */
typedef struct {
    size_t original_size;
    size_t compressed_size;
    double compression_ratio;
    uint32_t checksum;
    KolibriFileType file_type;
    uint32_t methods_used;
    double compression_time_ms;
    double decompression_time_ms;
} KolibriCompressStats;

/* Main compression context */
typedef struct KolibriCompressor KolibriCompressor;

/**
 * Create a new compressor instance
 * @param methods Bitfield of compression methods to use
 * @return New compressor instance or NULL on failure
 */
KolibriCompressor *kolibri_compressor_create(uint32_t methods);

/**
 * Destroy a compressor instance
 */
void kolibri_compressor_destroy(KolibriCompressor *comp);

/**
 * Compress data with automatic method selection
 * @param comp Compressor instance
 * @param input Input data
 * @param input_size Size of input data
 * @param output Output buffer (will be allocated)
 * @param output_size Size of output buffer
 * @param stats Optional statistics output
 * @return 0 on success, negative on error
 */
int kolibri_compress(KolibriCompressor *comp,
                     const uint8_t *input,
                     size_t input_size,
                     uint8_t **output,
                     size_t *output_size,
                     KolibriCompressStats *stats);

/**
 * Decompress data
 * @param input Compressed data
 * @param input_size Size of compressed data
 * @param output Output buffer (will be allocated)
 * @param output_size Size of output buffer
 * @param stats Optional statistics output
 * @return 0 on success, negative on error
 */
int kolibri_decompress(const uint8_t *input,
                       size_t input_size,
                       uint8_t **output,
                       size_t *output_size,
                       KolibriCompressStats *stats);

/**
 * Detect file type from data
 */
KolibriFileType kolibri_detect_file_type(const uint8_t *data, size_t size);

/**
 * Calculate checksum for data integrity
 */
uint32_t kolibri_checksum(const uint8_t *data, size_t size);

/* Archive management */
typedef struct KolibriArchive KolibriArchive;

typedef struct {
    char name[256];
    size_t original_size;
    size_t compressed_size;
    uint32_t checksum;
    uint64_t timestamp;
    KolibriFileType type;
} KolibriArchiveEntry;

/* ============================================================================
 * Streaming (incremental) API — сжатие/распаковка потоковых данных
 * ============================================================================ */

/** Тип операции потока */
typedef enum {
    KOLIBRI_STREAM_COMPRESS,
    KOLIBRI_STREAM_DECOMPRESS
} KolibriStreamMode;

/** Статус потоковой операции */
typedef enum {
    KOLIBRI_STREAM_OK         =  0,   /* Успех */
    KOLIBRI_STREAM_ERROR      = -1,   /* Внутренняя ошибка */
    KOLIBRI_STREAM_DONE       =  1,   /* Поток завершён */
    KOLIBRI_STREAM_NEED_MORE  =  2    /* Нужно больше входных данных */
} KolibriStreamStatus;

/** Колбэк вывода: вызывается потоком для записи готовых данных.
 *  @param user_data   Пользовательский контекст (файл, буфер, …)
 *  @param data        Указатель на выходные данные
 *  @param size        Размер выходных данных
 *  @return            0 = OK, иначе ошибка → поток прервётся
 */
typedef int (*KolibriStreamWriteFn)(void *user_data,
                                     const uint8_t *data, size_t size);

/** Opaque streaming context */
typedef struct KolibriStream KolibriStream;

/**
 * Создать поток сжатия/распаковки.
 * @param mode       KOLIBRI_STREAM_COMPRESS или KOLIBRI_STREAM_DECOMPRESS
 * @param methods    Битовые флаги методов (для сжатия; игнорируется при распаковке)
 * @param write_fn   Колбэк для записи выходных данных
 * @param user_data  Пользовательский контекст, передаваемый в write_fn
 * @return Новый контекст потока или NULL при ошибке
 */
KolibriStream *kolibri_stream_create(KolibriStreamMode mode,
                                      uint32_t methods,
                                      KolibriStreamWriteFn write_fn,
                                      void *user_data);

/**
 * Подать входные данные в поток.
 * Можно вызывать многократно, передавая произвольные порции данных.
 * Внутри накапливается блок KF62_BLOCK_SIZE, после чего он сжимается
 * (или распаковывается) и результат отправляется через write_fn.
 * @return KOLIBRI_STREAM_OK или KOLIBRI_STREAM_ERROR
 */
KolibriStreamStatus kolibri_stream_write(KolibriStream *stream,
                                          const uint8_t *data,
                                          size_t size);

/**
 * Завершить поток: сбросить оставшиеся данные, записать финализатор.
 * После вызова поток становится недействительным (нужно уничтожить).
 * @return KOLIBRI_STREAM_DONE или KOLIBRI_STREAM_ERROR
 */
KolibriStreamStatus kolibri_stream_finish(KolibriStream *stream);

/**
 * Получить промежуточную статистику потока.
 * @param stream   Контекст потока
 * @param stats    Заполняемая структура статистики
 */
void kolibri_stream_stats(const KolibriStream *stream,
                           KolibriCompressStats *stats);

/**
 * Уничтожить потоковый контекст и освободить ресурсы.
 */
void kolibri_stream_destroy(KolibriStream *stream);

/* ============================================================================
 * Archive management
 * ============================================================================ */

/**
 * Create a new archive
 */
KolibriArchive *kolibri_archive_create(const char *filename);

/**
 * Open an existing archive
 */
KolibriArchive *kolibri_archive_open(const char *filename);

/**
 * Add file to archive
 */
int kolibri_archive_add_file(KolibriArchive *archive,
                              const char *filename,
                              const uint8_t *data,
                              size_t size);

/**
 * Extract file from archive
 */
int kolibri_archive_extract_file(KolibriArchive *archive,
                                  const char *filename,
                                  uint8_t **data,
                                  size_t *size);

/**
 * List archive contents
 */
int kolibri_archive_list(KolibriArchive *archive,
                         KolibriArchiveEntry **entries,
                         size_t *count);

/**
 * Close and save archive
 */
void kolibri_archive_close(KolibriArchive *archive);

#ifdef __cplusplus
}
#endif

#endif /* KOLIBRI_COMPRESS_H */
