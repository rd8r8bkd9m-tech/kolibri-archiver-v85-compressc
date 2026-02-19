/*
 * Kolibri OS Archiver - Command Line Interface
 * Archive creation, extraction, and management tool
 */

#include "kolibri/compress.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>

static void print_usage(const char *prog) {
    printf("Kolibri OS Archiver v%d - Advanced Compression System\n\n",
           KOLIBRI_ARCHIVER_VERSION_MAJOR);
    printf("Usage: %s <command> [options]\n\n", prog);
    printf("Commands:\n");
    printf("  compress <input> <output>    Compress file or directory\n");
    printf("  decompress <input> <output>  Decompress file\n");
    printf("  create <archive>             Create new archive\n");
    printf("  add <archive> <file>         Add file to archive\n");
    printf("  extract <archive> <file>     Extract file from archive\n");
    printf("  list <archive>               List archive contents\n");
    printf("  info <file.kolibri>          Show archive information\n");
    printf("  test <input>                 Test compression roundtrip\n");
    printf("  bench <input>                Benchmark all methods\n");
    printf("  version                      Show version information\n");
    printf("\nOptions:\n");
    printf("  --help                       Show this help message\n");
    printf("\nExamples:\n");
    printf("  %s compress myfile.txt myfile.klb\n", prog);
    printf("  %s compress mydir/ archive.klb\n", prog);
    printf("  %s decompress myfile.klb myfile.txt\n", prog);
    printf("  %s create archive.kar\n", prog);
    printf("  %s add archive.kar document.pdf\n", prog);
    printf("  %s list archive.kar\n", prog);
    printf("  %s info myfile.klb\n", prog);
}

static uint8_t *read_file(const char *filename, size_t *size) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file '%s'\n", filename);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size < 0) {
        fclose(f);
        fprintf(stderr, "Error: Invalid file size\n");
        return NULL;
    }

    /* Пустые файлы — допустимый случай */
    if (file_size == 0) {
        fclose(f);
        *size = 0;
        uint8_t *empty = (uint8_t *)malloc(1); /* non-NULL sentinel */
        return empty;
    }

    uint8_t *buffer = (uint8_t *)malloc(file_size);
    if (!buffer) {
        fclose(f);
        fprintf(stderr, "Error: Memory allocation failed\n");
        return NULL;
    }

    if (fread(buffer, 1, file_size, f) != (size_t)file_size) {
        free(buffer);
        fclose(f);
        fprintf(stderr, "Error: Failed to read file\n");
        return NULL;
    }

    fclose(f);
    *size = file_size;
    return buffer;
}

static int write_file(const char *filename, const uint8_t *data, size_t size) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error: Cannot create file '%s'\n", filename);
        return -1;
    }

    if (fwrite(data, 1, size, f) != size) {
        fclose(f);
        fprintf(stderr, "Error: Failed to write file\n");
        return -1;
    }

    fclose(f);
    return 0;
}

static void print_file_type(KolibriFileType type) {
    switch (type) {
        case KOLIBRI_FILE_TEXT:
            printf("Text");
            break;
        case KOLIBRI_FILE_BINARY:
            printf("Binary");
            break;
        case KOLIBRI_FILE_IMAGE:
            printf("Image");
            break;
        default:
            printf("Unknown");
            break;
    }
}

static void print_methods(uint32_t methods) {
    int first = 1;
    if (methods & KOLIBRI_COMPRESS_MATH) {
        printf("Mathematical");
        first = 0;
    }
    if (methods & KOLIBRI_COMPRESS_LZ77) {
        if (!first) printf("+");
        printf("LZ77");
        first = 0;
    }
    if (methods & KOLIBRI_COMPRESS_RLE) {
        if (!first) printf("+");
        printf("RLE");
        first = 0;
    }
    if (methods & KOLIBRI_COMPRESS_HUFFMAN) {
        if (!first) printf("+");
        printf("Huffman");
        first = 0;
    }
    if (methods & KOLIBRI_COMPRESS_FORMULA) {
        if (!first) printf("+");
        printf("Formula");
        first = 0;
    }
    if (methods & KOLIBRI_COMPRESS_TOKEN) {
        if (!first) printf("+");
        printf("Token");
        first = 0;
    }
    if (methods & KOLIBRI_COMPRESS_LZCM) {
        if (!first) printf("+");
        printf("LZCM");
        first = 0;
    }
    if (first) {
        printf("Raw (stored)");
    }
}

static int cmd_compress(const char *input, const char *output) {
    printf("Compressing '%s' to '%s'...\n", input, output);

    size_t input_size;
    uint8_t *input_data = read_file(input, &input_size);
    if (!input_data) {
        return 1;
    }

    KolibriCompressor *comp = kolibri_compressor_create(KOLIBRI_COMPRESS_ALL);
    if (!comp) {
        free(input_data);
        fprintf(stderr, "Error: Failed to create compressor\n");
        return 1;
    }

    uint8_t *output_data = NULL;
    size_t output_size = 0;
    KolibriCompressStats stats;

    int ret = kolibri_compress(comp, input_data, input_size, 
                                &output_data, &output_size, &stats);
    kolibri_compressor_destroy(comp);
    free(input_data);

    if (ret != 0) {
        fprintf(stderr, "Error: Compression failed\n");
        return 1;
    }

    ret = write_file(output, output_data, output_size);
    free(output_data);

    if (ret != 0) {
        return 1;
    }

    printf("\nCompression complete!\n");
    printf("Original size:    %zu bytes\n", stats.original_size);
    printf("Compressed size:  %zu bytes\n", stats.compressed_size);
    printf("Compression ratio: %.2fx\n", stats.compression_ratio);
    printf("File type:        ");
    print_file_type(stats.file_type);
    printf("\nMethods used:     ");
    print_methods(stats.methods_used);
    printf("\nCompression time: %.2f ms\n", stats.compression_time_ms);
    printf("Checksum:         0x%08X\n", stats.checksum);

    return 0;
}

static int cmd_decompress(const char *input, const char *output) {
    printf("Decompressing '%s' to '%s'...\n", input, output);

    size_t input_size;
    uint8_t *input_data = read_file(input, &input_size);
    if (!input_data) {
        return 1;
    }

    printf("Read %zu bytes from input file\n", input_size);

    uint8_t *output_data = NULL;
    size_t output_size = 0;
    KolibriCompressStats stats;

    int ret = kolibri_decompress(input_data, input_size, 
                                  &output_data, &output_size, &stats);
    free(input_data);

    if (ret != 0) {
        fprintf(stderr, "Error: Decompression failed with code %d\n", ret);
        return 1;
    }

    ret = write_file(output, output_data, output_size);
    free(output_data);

    if (ret != 0) {
        return 1;
    }

    printf("\nDecompression complete!\n");
    printf("Compressed size:   %zu bytes\n", stats.compressed_size);
    printf("Decompressed size: %zu bytes\n", stats.original_size);
    printf("Compression ratio: %.2fx\n", stats.compression_ratio);
    printf("Decompression time: %.2f ms\n", stats.decompression_time_ms);
    printf("Checksum verified: 0x%08X\n", stats.checksum);

    return 0;
}

static int cmd_create(const char *archive_name) {
    printf("Creating archive '%s'...\n", archive_name);

    KolibriArchive *archive = kolibri_archive_create(archive_name);
    if (!archive) {
        fprintf(stderr, "Error: Failed to create archive\n");
        return 1;
    }

    kolibri_archive_close(archive);
    printf("Archive created successfully.\n");

    return 0;
}

static int cmd_add(const char *archive_name, const char *filename) {
    printf("Adding '%s' to archive '%s'...\n", filename, archive_name);

    KolibriArchive *archive = kolibri_archive_open(archive_name);
    if (!archive) {
        fprintf(stderr, "Error: Cannot open archive '%s'\n", archive_name);
        return 1;
    }

    size_t file_size;
    uint8_t *file_data = read_file(filename, &file_size);
    if (!file_data) {
        kolibri_archive_close(archive);
        return 1;
    }

    int ret = kolibri_archive_add_file(archive, filename, file_data, file_size);
    free(file_data);
    kolibri_archive_close(archive);

    if (ret != 0) {
        fprintf(stderr, "Error: Failed to add file to archive\n");
        return 1;
    }

    printf("File added successfully.\n");
    return 0;
}

static int cmd_extract(const char *archive_name, const char *filename) {
    printf("Extracting '%s' from archive '%s'...\n", filename, archive_name);

    KolibriArchive *archive = kolibri_archive_open(archive_name);
    if (!archive) {
        fprintf(stderr, "Error: Cannot open archive '%s'\n", archive_name);
        return 1;
    }

    uint8_t *file_data = NULL;
    size_t file_size = 0;

    int ret = kolibri_archive_extract_file(archive, filename, &file_data, &file_size);
    kolibri_archive_close(archive);

    if (ret != 0) {
        fprintf(stderr, "Error: Failed to extract file from archive\n");
        return 1;
    }

    ret = write_file(filename, file_data, file_size);
    free(file_data);

    if (ret != 0) {
        return 1;
    }

    printf("File extracted successfully.\n");
    return 0;
}

static int cmd_list(const char *archive_name) {
    printf("Listing contents of archive '%s':\n\n", archive_name);

    KolibriArchive *archive = kolibri_archive_open(archive_name);
    if (!archive) {
        fprintf(stderr, "Error: Cannot open archive '%s'\n", archive_name);
        return 1;
    }

    KolibriArchiveEntry *entries = NULL;
    size_t count = 0;

    int ret = kolibri_archive_list(archive, &entries, &count);
    kolibri_archive_close(archive);

    if (ret != 0) {
        fprintf(stderr, "Error: Failed to list archive contents\n");
        return 1;
    }

    if (count == 0) {
        printf("Archive is empty.\n");
        return 0;
    }

    printf("%-40s %12s %12s %8s %s\n", 
           "Name", "Original", "Compressed", "Ratio", "Type");
    printf("--------------------------------------------------------------------------------\n");

    for (size_t i = 0; i < count; i++) {
        double ratio = (double)entries[i].original_size / (double)entries[i].compressed_size;
        printf("%-40s %12zu %12zu %7.2fx ", 
               entries[i].name,
               entries[i].original_size,
               entries[i].compressed_size,
               ratio);
        print_file_type(entries[i].type);
        printf("\n");
    }

    printf("--------------------------------------------------------------------------------\n");
    printf("Total files: %zu\n", count);

    free(entries);
    return 0;
}

static int cmd_version(void) {
    printf("Kolibri OS Archiver\n");
    printf("Version: v%d.%d.%d\n",
           KOLIBRI_ARCHIVER_VERSION_MAJOR,
           KOLIBRI_ARCHIVER_VERSION_MINOR,
           KOLIBRI_ARCHIVER_VERSION_PATCH);
    printf("Build date: %s %s\n", __DATE__, __TIME__);
    printf("\nМетоды сжатия:\n");
    printf("  - LZCM     LZ + Context Mixing (v66, основной)\n");
    printf("  - LZ-lite  Dictionary-based (hash chains, rep-match)\n");
    printf("  - Formula  15-predictor CM v62 (SSE/APM, SIMD)\n");
    printf("  - Token    Text stream tokenisation (v52)\n");
    printf("  - Huffman  Entropy coding (tANS)\n");
    printf("  - RLE      Run-Length Encoding\n");
    printf("  - Math     BWT + MTF + ZRLE + Delta + Generator\n");
    printf("\nВозможности:\n");
    printf("  - Многослойное сжатие (LZ → CM pipeline)\n");
    printf("  - Автоопределение типа файла (Text/Binary/Image)\n");
    printf("  - UTF-8 / кириллица / CJK текстовая токенизация\n");
    printf("  - CRC32 контроль целостности\n");
    printf("  - Многофайловые архивы KARC (до 1024 файлов)\n");
    printf("  - Многопоточное сжатие (до 4 потоков)\n");
    printf("  - Потоковый API (streaming)\n");
    printf("  - Поддержка пустых файлов и любых бинарных данных\n");
    return 0;
}

static int cmd_test(const char *input) {
    printf("Testing compression on '%s'...\n", input);

    size_t input_size;
    uint8_t *input_data = read_file(input, &input_size);
    if (!input_data) {
        return 1;
    }

    KolibriCompressor *comp = kolibri_compressor_create(KOLIBRI_COMPRESS_ALL);
    if (!comp) {
        free(input_data);
        fprintf(stderr, "Error: Failed to create compressor\n");
        return 1;
    }

    uint8_t *compressed_data = NULL;
    size_t compressed_size = 0;
    KolibriCompressStats stats;

    int ret = kolibri_compress(comp, input_data, input_size, 
                                &compressed_data, &compressed_size, &stats);
    if (ret != 0) {
        kolibri_compressor_destroy(comp);
        free(input_data);
        fprintf(stderr, "Error: Compression failed\n");
        return 1;
    }

    /* Test decompression */
    uint8_t *decompressed_data = NULL;
    size_t decompressed_size = 0;

    ret = kolibri_decompress(compressed_data, compressed_size,
                             &decompressed_data, &decompressed_size, NULL);
    free(compressed_data);

    if (ret != 0) {
        kolibri_compressor_destroy(comp);
        free(input_data);
        fprintf(stderr, "Error: Decompression failed\n");
        return 1;
    }

    /* Verify integrity */
    int match = (decompressed_size == input_size &&
                 memcmp(input_data, decompressed_data, input_size) == 0);

    free(input_data);
    free(decompressed_data);
    kolibri_compressor_destroy(comp);

    printf("\nTest Results:\n");
    printf("Original size:     %zu bytes\n", stats.original_size);
    printf("Compressed size:   %zu bytes\n", stats.compressed_size);
    printf("Compression ratio: %.2fx\n", stats.compression_ratio);
    printf("File type:         ");
    print_file_type(stats.file_type);
    printf("\nMethods used:      ");
    print_methods(stats.methods_used);
    printf("\nCompression time:  %.2f ms\n", stats.compression_time_ms);
    printf("Data integrity:    %s\n", match ? "PASSED ✓" : "FAILED ✗");

    return match ? 0 : 1;
}

/* ============================================================================
 * info — показать информацию о .kolibri файле без распаковки
 * ============================================================================ */
static int cmd_info(const char *input) {
    size_t input_size;
    uint8_t *input_data = read_file(input, &input_size);
    if (!input_data) return 1;

    printf("Файл: %s (%zu байт)\n\n", input, input_size);

    if (input_size < 5) {
        printf("Слишком маленький файл для распознавания формата.\n");
        free(input_data);
        return 1;
    }

    /* Проверяем минимальный LZCM заголовок */
    uint16_t first_magic;
    memcpy(&first_magic, input_data, 2);
    if (first_magic == 0x4D4B) { /* "KM" */
        uint32_t orig = (uint32_t)input_data[2]
                      | ((uint32_t)input_data[3] << 8)
                      | ((uint32_t)input_data[4] << 16);
        printf("Формат:     LZCM (минимальный заголовок)\n");
        printf("Оригинал:   %u байт\n", orig);
        printf("Сжатие:     %zu байт\n", input_size);
        if (orig > 0)
            printf("Степень:    %.2fx\n", (double)orig / (double)input_size);
        printf("Метод:      LZCM v66 (LZ + Context Mixing)\n");
        free(input_data);
        return 0;
    }

    /* Традиционный KLBR заголовок */
    if (input_size >= 16) {
        uint32_t magic;
        memcpy(&magic, input_data, 4);
        if (magic == 0x4B4C4252) { /* KLBR */
            uint16_t ver, methods;
            uint32_t orig_size, checksum;
            memcpy(&ver, input_data + 4, 2);
            memcpy(&methods, input_data + 6, 2);
            memcpy(&orig_size, input_data + 8, 4);
            memcpy(&checksum, input_data + 12, 4);

            printf("Формат:     KLBR (традиционный)\n");
            printf("Версия:     v%u\n", ver);
            printf("Оригинал:   %u байт\n", orig_size);
            printf("Сжатие:     %zu байт\n", input_size);
            if (orig_size > 0)
                printf("Степень:    %.2fx\n", (double)orig_size / (double)input_size);
            printf("CRC32:      0x%08X\n", checksum);
            printf("Методы:     ");
            print_methods(methods);
            printf("\n");
            free(input_data);
            return 0;
        }

        /* KARC архив */
        if (magic == 0x4B415243) {
            uint32_t arc_ver, entry_count;
            memcpy(&arc_ver, input_data + 4, 4);
            memcpy(&entry_count, input_data + 8, 4);
            printf("Формат:     KARC (многофайловый архив)\n");
            printf("Версия:     v%u\n", arc_ver);
            printf("Файлов:     %u\n", entry_count);
            printf("Размер:     %zu байт\n", input_size);
            free(input_data);
            return 0;
        }
    }

    printf("Неизвестный формат файла.\n");
    free(input_data);
    return 1;
}

/* ============================================================================
 * bench — сравнение всех методов на одном файле
 * ============================================================================ */
static int cmd_bench(const char *input) {
    printf("Бенчмарк сжатия: '%s'\n\n", input);

    size_t input_size;
    uint8_t *input_data = read_file(input, &input_size);
    if (!input_data) return 1;

    if (input_size == 0) {
        printf("Пустой файл — бенчмарк невозможен.\n");
        free(input_data);
        return 0;
    }

    KolibriFileType ft = kolibri_detect_file_type(input_data, input_size);
    printf("Тип файла: ");
    print_file_type(ft);
    printf("\nРазмер:    %zu байт\n\n", input_size);

    printf("%-20s %12s %8s %10s %s\n",
           "Метод", "Размер", "Степень", "Время (мс)", "Целостность");
    printf("------------------------------------------------------------------------\n");

    /* Тест каждого метода отдельно */
    struct { const char *name; uint32_t method; } methods[] = {
        {"LZCM (unified)",   KOLIBRI_COMPRESS_ALL},
        {"LZ-lite only",     KOLIBRI_COMPRESS_LZ77},
        {"Formula CM v62",   KOLIBRI_COMPRESS_FORMULA},
        {"LZ+Formula",       KOLIBRI_COMPRESS_LZ77 | KOLIBRI_COMPRESS_FORMULA},
    };
    int n_methods = (int)(sizeof(methods) / sizeof(methods[0]));

    for (int m = 0; m < n_methods; m++) {
        KolibriCompressor *comp = kolibri_compressor_create(methods[m].method);
        if (!comp) continue;

        uint8_t *cdata = NULL;
        size_t csize = 0;
        KolibriCompressStats st;

        int ret = kolibri_compress(comp, input_data, input_size, &cdata, &csize, &st);
        kolibri_compressor_destroy(comp);

        if (ret != 0) {
            printf("%-20s %12s %8s %10s %s\n",
                   methods[m].name, "-", "-", "-", "ОШИБКА");
            continue;
        }

        /* Roundtrip */
        uint8_t *ddata = NULL;
        size_t dsize = 0;
        ret = kolibri_decompress(cdata, csize, &ddata, &dsize, NULL);
        int ok = (ret == 0 && dsize == input_size &&
                  memcmp(input_data, ddata, input_size) == 0);
        free(cdata);
        free(ddata);

        printf("%-20s %12zu %7.2fx %10.1f %s\n",
               methods[m].name, st.compressed_size, st.compression_ratio,
               st.compression_time_ms, ok ? "✓" : "FAIL ✗");
    }

    printf("------------------------------------------------------------------------\n");
    free(input_data);
    return 0;
}

/* ============================================================================
 * compress directory — рекурсивно добавить все файлы в KARC архив
 * ============================================================================ */
static int add_directory_recursive(KolibriArchive *archive,
                                    const char *dir_path,
                                    const char *prefix,
                                    int *total_files,
                                    size_t *total_original,
                                    size_t *total_compressed) {
    DIR *d = opendir(dir_path);
    if (!d) {
        fprintf(stderr, "Ошибка: не удалось открыть директорию '%s'\n", dir_path);
        return -1;
    }

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
            continue;

        char full_path[1024];
        char arc_name[512];
        snprintf(full_path, sizeof(full_path), "%s/%s", dir_path, ent->d_name);
        if (prefix[0])
            snprintf(arc_name, sizeof(arc_name), "%s/%s", prefix, ent->d_name);
        else
            snprintf(arc_name, sizeof(arc_name), "%s", ent->d_name);

        struct stat st;
        if (stat(full_path, &st) != 0) continue;

        if (S_ISDIR(st.st_mode)) {
            /* Рекурсия в поддиректорию */
            add_directory_recursive(archive, full_path, arc_name,
                                    total_files, total_original, total_compressed);
        } else if (S_ISREG(st.st_mode)) {
            size_t fsize;
            uint8_t *fdata = read_file(full_path, &fsize);
            if (!fdata) continue;

            int ret = kolibri_archive_add_file(archive, arc_name, fdata, fsize);
            free(fdata);

            if (ret == 0) {
                (*total_files)++;
                *total_original += fsize;
                printf("  + %-50s %8zu байт\n", arc_name, fsize);
            } else {
                fprintf(stderr, "  ! Ошибка: '%s'\n", arc_name);
            }
        }
    }
    closedir(d);
    return 0;
}

static int cmd_compress_dir(const char *dir_path, const char *output) {
    printf("Сжатие директории '%s' → '%s'...\n\n", dir_path, output);

    KolibriArchive *archive = kolibri_archive_create(output);
    if (!archive) {
        fprintf(stderr, "Ошибка: не удалось создать архив '%s'\n", output);
        return 1;
    }

    int total_files = 0;
    size_t total_original = 0;
    size_t total_compressed = 0;

    int ret = add_directory_recursive(archive, dir_path, "",
                                       &total_files, &total_original, &total_compressed);
    kolibri_archive_close(archive);

    if (ret != 0) return 1;

    /* Показать итоги */
    struct stat st;
    size_t archive_size = 0;
    if (stat(output, &st) == 0) archive_size = (size_t)st.st_size;

    printf("\nАрхив создан: %s\n", output);
    printf("Файлов:       %d\n", total_files);
    printf("Оригинал:     %zu байт\n", total_original);
    printf("Архив:        %zu байт\n", archive_size);
    if (total_original > 0)
        printf("Степень:      %.2fx\n", (double)total_original / (double)archive_size);

    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char *cmd = argv[1];

    if (strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) {
        print_usage(argv[0]);
        return 0;
    }

    if (strcmp(cmd, "compress") == 0) {
        if (argc != 4) {
            fprintf(stderr, "Usage: %s compress <input> <output>\n", argv[0]);
            return 1;
        }
        /* Проверяем: входной путь — директория или файл? */
        struct stat st;
        if (stat(argv[2], &st) == 0 && S_ISDIR(st.st_mode)) {
            return cmd_compress_dir(argv[2], argv[3]);
        }
        return cmd_compress(argv[2], argv[3]);
    }

    if (strcmp(cmd, "decompress") == 0) {
        if (argc != 4) {
            fprintf(stderr, "Usage: %s decompress <input> <output>\n", argv[0]);
            return 1;
        }
        return cmd_decompress(argv[2], argv[3]);
    }

    if (strcmp(cmd, "create") == 0) {
        if (argc != 3) {
            fprintf(stderr, "Usage: %s create <archive>\n", argv[0]);
            return 1;
        }
        return cmd_create(argv[2]);
    }

    if (strcmp(cmd, "add") == 0) {
        if (argc != 4) {
            fprintf(stderr, "Usage: %s add <archive> <file>\n", argv[0]);
            return 1;
        }
        return cmd_add(argv[2], argv[3]);
    }

    if (strcmp(cmd, "extract") == 0) {
        if (argc != 4) {
            fprintf(stderr, "Usage: %s extract <archive> <file>\n", argv[0]);
            return 1;
        }
        return cmd_extract(argv[2], argv[3]);
    }

    if (strcmp(cmd, "list") == 0) {
        if (argc != 3) {
            fprintf(stderr, "Usage: %s list <archive>\n", argv[0]);
            return 1;
        }
        return cmd_list(argv[2]);
    }

    if (strcmp(cmd, "test") == 0) {
        if (argc != 3) {
            fprintf(stderr, "Usage: %s test <input>\n", argv[0]);
            return 1;
        }
        return cmd_test(argv[2]);
    }

    if (strcmp(cmd, "info") == 0) {
        if (argc != 3) {
            fprintf(stderr, "Usage: %s info <file>\n", argv[0]);
            return 1;
        }
        return cmd_info(argv[2]);
    }

    if (strcmp(cmd, "bench") == 0) {
        if (argc != 3) {
            fprintf(stderr, "Usage: %s bench <input>\n", argv[0]);
            return 1;
        }
        return cmd_bench(argv[2]);
    }

    if (strcmp(cmd, "version") == 0 || strcmp(cmd, "-v") == 0 || strcmp(cmd, "--version") == 0) {
        return cmd_version();
    }

    fprintf(stderr, "Unknown command: %s\n", cmd);
    print_usage(argv[0]);
    return 1;
}
