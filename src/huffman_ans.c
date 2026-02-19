/*
 * huffman_ans.c — Huffman + ANS (Asymmetric Numeral Systems) для Kolibri
 *
 * Два режима сжатия для конкуренции с gzip на общих данных:
 *   1. Huffman — классическое prefix-free кодирование
 *   2. tANS — табличный ANS (быстрее Huffman, сжатие как arithmetic)
 *
 * Интеграция с архиватором Kolibri:
 *   - Вызывается как альтернативный бэкенд через kolibri_compress()
 *   - Совместим с existing Mathematical+LZ77+RLE pipeline
 *
 * Copyright (c) 2025 Кочуров Владислав Евгеньевич
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * Huffman Coding
 * ============================================================================ */

#define HUFFMAN_MAX_SYMBOLS 256
#define HUFFMAN_MAX_CODE_LEN 32

/* Узел дерева Хаффмана */
typedef struct HuffNode {
    uint64_t freq;
    int symbol;      /* -1 для внутренних узлов */
    struct HuffNode *left;
    struct HuffNode *right;
} HuffNode;

/* Таблица кодов */
typedef struct {
    uint32_t code;
    uint8_t  length;
} HuffCode;

/* Контекст Huffman */
typedef struct {
    HuffCode codes[HUFFMAN_MAX_SYMBOLS];
    uint64_t freq[HUFFMAN_MAX_SYMBOLS];
    HuffNode *tree_root;
    HuffNode  nodes[HUFFMAN_MAX_SYMBOLS * 2]; /* Пул узлов */
    size_t    node_count;
} HuffmanContext;

/* --- Внутренние функции --- */

static HuffNode *huff_alloc_node(HuffmanContext *ctx) {
    if (ctx->node_count >= HUFFMAN_MAX_SYMBOLS * 2) return NULL;
    HuffNode *n = &ctx->nodes[ctx->node_count++];
    n->freq = 0;
    n->symbol = -1;
    n->left = NULL;
    n->right = NULL;
    return n;
}

static void huff_build_codes(HuffmanContext *ctx, HuffNode *node,
                              uint32_t code, uint8_t depth) {
    if (!node) return;
    if (node->symbol >= 0) {
        /* Лист */
        ctx->codes[node->symbol].code = code;
        ctx->codes[node->symbol].length = depth > 0 ? depth : 1;
        return;
    }
    if (depth >= HUFFMAN_MAX_CODE_LEN) return;
    huff_build_codes(ctx, node->left,  (code << 1) | 0, depth + 1);
    huff_build_codes(ctx, node->right, (code << 1) | 1, depth + 1);
}

static void huff_init(HuffmanContext *ctx) {
    memset(ctx, 0, sizeof(*ctx));
}

static int huff_build_tree(HuffmanContext *ctx) {
    /* Min-heap через массив указателей */
    HuffNode *heap[HUFFMAN_MAX_SYMBOLS];
    int heap_size = 0;

    /* Создаём листья для символов с ненулевой частотой */
    for (int i = 0; i < HUFFMAN_MAX_SYMBOLS; ++i) {
        if (ctx->freq[i] > 0) {
            HuffNode *leaf = huff_alloc_node(ctx);
            if (!leaf) return -1;
            leaf->freq = ctx->freq[i];
            leaf->symbol = i;
            heap[heap_size++] = leaf;
        }
    }

    if (heap_size == 0) return -1;
    if (heap_size == 1) {
        /* Один символ — код = 0, длина = 1 */
        ctx->tree_root = heap[0];
        ctx->codes[heap[0]->symbol].code = 0;
        ctx->codes[heap[0]->symbol].length = 1;
        return 0;
    }

    /* Сортировка (простая — до 256 элементов) */
    for (int i = 0; i < heap_size - 1; ++i) {
        for (int j = i + 1; j < heap_size; ++j) {
            if (heap[j]->freq < heap[i]->freq) {
                HuffNode *tmp = heap[i];
                heap[i] = heap[j];
                heap[j] = tmp;
            }
        }
    }

    /* Строим дерево */
    while (heap_size > 1) {
        HuffNode *left = heap[0];
        HuffNode *right = heap[1];

        HuffNode *parent = huff_alloc_node(ctx);
        if (!parent) return -1;
        parent->freq = left->freq + right->freq;
        parent->left = left;
        parent->right = right;

        /* Удаляем два минимальных, вставляем parent */
        heap_size -= 2;
        for (int i = 0; i < heap_size; ++i) heap[i] = heap[i + 2];
        
        /* Вставка с сохранением порядка */
        int pos = heap_size;
        for (int i = 0; i < heap_size; ++i) {
            if (parent->freq <= heap[i]->freq) {
                pos = i;
                break;
            }
        }
        for (int i = heap_size; i > pos; --i) heap[i] = heap[i - 1];
        heap[pos] = parent;
        heap_size++;
    }

    ctx->tree_root = heap[0];
    huff_build_codes(ctx, ctx->tree_root, 0, 0);

    /* ПРЕОБРАЗОВАНИЕ В КАНОНИЧЕСКИЕ КОДЫ */
    /* Важно: decoder восстанавливает код только по длинам (Canonical Huffman). */
    /* Поэтому encoder тоже должен использовать канонические коды, а не те, */
    /* что получились при случайном обходе дерева. */

    int symbols_by_len[256];
    int sym_count = 0;
    for (int i = 0; i < HUFFMAN_MAX_SYMBOLS; ++i) {
        if (ctx->codes[i].length > 0) {
            symbols_by_len[sym_count++] = i;
        }
    }

    /* Сортировка: по длине (возр), затем лексикографически */
    for (int i = 0; i < sym_count - 1; ++i) {
        for (int j = i + 1; j < sym_count; ++j) {
            int si = symbols_by_len[i];
            int sj = symbols_by_len[j];
            if (ctx->codes[sj].length < ctx->codes[si].length ||
               (ctx->codes[sj].length == ctx->codes[si].length && sj < si)) {
                int tmp = symbols_by_len[i];
                symbols_by_len[i] = symbols_by_len[j];
                symbols_by_len[j] = tmp;
            }
        }
    }

    /* Назначение канонических кодов */
    uint32_t c_code = 0;
    uint8_t prev_len = 0;
    for (int i = 0; i < sym_count; ++i) {
        int sym = symbols_by_len[i];
        uint8_t len = ctx->codes[sym].length;
        if (prev_len > 0) {
            c_code = (c_code + 1) << (len - prev_len);
        }
        ctx->codes[sym].code = c_code;
        ctx->codes[sym].length = len; /* Длина остаётся той же */
        prev_len = len;
    }

    return 0;
}

/* --- Публичное API --- */

/*
 * huffman_compress — сжатие буфера Huffman-кодированием
 *
 * Три формата таблицы (автовыбор наименьшего):
 *   KHU2 (sparse): [4]magic [4]size [1]N [N]syms [⌈N/2⌉]packed_lens
 *   KHU3 (packed256): [4]magic [4]size [1]base [128]packed_offsets
 *   KHUF (full):  [4]magic [4]size [256]raw_lengths
 *
 * Returns: размер сжатых данных, 0 при ошибке
 */
size_t huffman_compress(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_capacity
) {
    if (!input || !output || input_size == 0) return 0;

    HuffmanContext ctx;
    huff_init(&ctx);

    /* Подсчёт частот */
    for (size_t i = 0; i < input_size; ++i) {
        ctx.freq[input[i]]++;
    }

    /* Построение дерева */
    if (huff_build_tree(&ctx) != 0) return 0;

    /* --- Анализ уникальных символов для выбора формата --- */
    int sorted_syms[256];
    int num_unique = 0;
    uint8_t max_code_len = 0;
    uint8_t min_code_len = 255;
    for (int i = 0; i < HUFFMAN_MAX_SYMBOLS; ++i) {
        if (ctx.codes[i].length > 0) {
            sorted_syms[num_unique++] = i;
            if (ctx.codes[i].length > max_code_len)
                max_code_len = ctx.codes[i].length;
            if (ctx.codes[i].length < min_code_len)
                min_code_len = ctx.codes[i].length;
        }
    }

    /* Сортировка по (длина, значение) — тот же порядок, что в canonical codes */
    for (int i = 0; i < num_unique - 1; ++i) {
        for (int j = i + 1; j < num_unique; ++j) {
            int si = sorted_syms[i], sj = sorted_syms[j];
            if (ctx.codes[sj].length < ctx.codes[si].length ||
               (ctx.codes[sj].length == ctx.codes[si].length && sj < si)) {
                int tmp = sorted_syms[i];
                sorted_syms[i] = sorted_syms[j];
                sorted_syms[j] = tmp;
            }
        }
    }

    /* --- Вычисляем размер заголовка каждого формата --- */
    size_t khu2_hdr = 9 + (size_t)num_unique + (size_t)(num_unique + 1) / 2;
    size_t khu3_hdr = 137;   /* 4+4+1+128 = всегда 137 */
    size_t khuf_hdr = 264;   /* 4+4+256 = всегда 264 */

    /* KHU2 (sparse): длины ≤ 15 */
    int can_khu2 = (max_code_len <= 15);
    /* KHU3 (packed256): диапазон длин помещается в 4 бита с 0=unused */
    /* offset = 0 → unused, 1..15 → length = base + offset */
    /* base = min_code_len - 1, max offset = max_code_len - base = max_code_len - min_code_len + 1 ≤ 15 */
    int can_khu3 = (min_code_len >= 1 && (max_code_len - min_code_len + 1) <= 15);

    /* Выбираем наименьший формат */
    int format = 0; /* 0=KHUF, 2=KHU2, 3=KHU3 */
    size_t best_hdr = khuf_hdr;
    if (can_khu3 && khu3_hdr < best_hdr) {
        best_hdr = khu3_hdr;
        format = 3;
    }
    if (can_khu2 && khu2_hdr < best_hdr) {
        best_hdr = khu2_hdr;
        format = 2;
    }

    size_t pos = 0;

    if (format == 2) {
        /* --- KHU2: компактный sparse --- */
        if (output_capacity < khu2_hdr + 1) return 0;

        output[pos++] = 'K'; output[pos++] = 'H';
        output[pos++] = 'U'; output[pos++] = '2';

        output[pos++] = (uint8_t)(input_size & 0xFF);
        output[pos++] = (uint8_t)((input_size >> 8) & 0xFF);
        output[pos++] = (uint8_t)((input_size >> 16) & 0xFF);
        output[pos++] = (uint8_t)((input_size >> 24) & 0xFF);

        output[pos++] = (uint8_t)num_unique;

        for (int i = 0; i < num_unique; ++i)
            output[pos++] = (uint8_t)sorted_syms[i];

        for (int i = 0; i < num_unique; i += 2) {
            uint8_t hi = ctx.codes[sorted_syms[i]].length & 0x0F;
            uint8_t lo = (i + 1 < num_unique)
                       ? (ctx.codes[sorted_syms[i + 1]].length & 0x0F) : 0;
            output[pos++] = (uint8_t)((hi << 4) | lo);
        }
    } else if (format == 3) {
        /* --- KHU3: packed 4-bit для всех 256 символов --- */
        if (output_capacity < khu3_hdr + 1) return 0;

        output[pos++] = 'K'; output[pos++] = 'H';
        output[pos++] = 'U'; output[pos++] = '3';

        output[pos++] = (uint8_t)(input_size & 0xFF);
        output[pos++] = (uint8_t)((input_size >> 8) & 0xFF);
        output[pos++] = (uint8_t)((input_size >> 16) & 0xFF);
        output[pos++] = (uint8_t)((input_size >> 24) & 0xFF);

        /* base = min_code_len - 1, offset 0 = unused, 1..15 = length - base */
        uint8_t base = min_code_len - 1;
        output[pos++] = base;

        for (int i = 0; i < 256; i += 2) {
            uint8_t hi = (ctx.codes[i].length > 0)
                       ? (uint8_t)(ctx.codes[i].length - base) : 0;
            uint8_t lo = (ctx.codes[i + 1].length > 0)
                       ? (uint8_t)(ctx.codes[i + 1].length - base) : 0;
            output[pos++] = (uint8_t)((hi << 4) | lo);
        }
    } else {
        /* --- KHUF: полный формат --- */
        if (output_capacity < khuf_hdr + 1) return 0;

        output[pos++] = 'K'; output[pos++] = 'H';
        output[pos++] = 'U'; output[pos++] = 'F';

        output[pos++] = (uint8_t)(input_size & 0xFF);
        output[pos++] = (uint8_t)((input_size >> 8) & 0xFF);
        output[pos++] = (uint8_t)((input_size >> 16) & 0xFF);
        output[pos++] = (uint8_t)((input_size >> 24) & 0xFF);

        for (int i = 0; i < 256; ++i)
            output[pos++] = ctx.codes[i].length;
    }

    /* --- Кодирование битового потока (одинаково для всех форматов) --- */
    uint32_t bit_buffer = 0;
    int bits_in_buffer = 0;

    for (size_t i = 0; i < input_size; ++i) {
        uint8_t sym = input[i];
        uint32_t code = ctx.codes[sym].code;
        uint8_t code_len = ctx.codes[sym].length;
        if (code_len == 0) code_len = 1;  /* Safety */

        for (int b = code_len - 1; b >= 0; --b) {
            bit_buffer = (bit_buffer << 1) | ((code >> b) & 1);
            bits_in_buffer++;
            if (bits_in_buffer == 8) {
                if (pos >= output_capacity) return 0;
                output[pos++] = (uint8_t)bit_buffer;
                bit_buffer = 0;
                bits_in_buffer = 0;
            }
        }
    }

    if (bits_in_buffer > 0) {
        bit_buffer <<= (8 - bits_in_buffer);
        if (pos >= output_capacity) return 0;
        output[pos++] = (uint8_t)bit_buffer;
    }

    return pos;
}

/*
 * huffman_decompress — декомпрессия Huffman (форматы: KHU2, KHU3, KHUF)
 *
 * Returns: размер восстановленных данных, 0 при ошибке
 */
size_t huffman_decompress(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_capacity
) {
    if (!input || !output || input_size < 9) return 0;

    /* Определяем формат по magic */
    if (input[0] != 'K' || input[1] != 'H' || input[2] != 'U') return 0;
    char fmt = input[3];
    if (fmt != '2' && fmt != '3' && fmt != 'F') return 0;

    /* Original size */
    uint32_t orig_size = (uint32_t)input[4]
                       | ((uint32_t)input[5] << 8)
                       | ((uint32_t)input[6] << 16)
                       | ((uint32_t)input[7] << 24);
    if (orig_size > output_capacity) return 0;

    /* --- Восстановление длин кодов --- */
    uint8_t code_lengths[256];
    memset(code_lengths, 0, sizeof(code_lengths));
    size_t data_offset;

    if (fmt == '2') {
        /* KHU2: компактный sparse формат */
        int num_unique = input[8];
        if (num_unique == 0) num_unique = 256;

        size_t hdr_end = 9 + (size_t)num_unique + (size_t)(num_unique + 1) / 2;
        if (input_size < hdr_end) return 0;

        size_t len_off = 9 + (size_t)num_unique;
        for (int i = 0; i < num_unique; i += 2) {
            uint8_t packed = input[len_off + i / 2];
            code_lengths[input[9 + i]] = (packed >> 4) & 0x0F;
            if (i + 1 < num_unique)
                code_lengths[input[9 + i + 1]] = packed & 0x0F;
        }
        data_offset = hdr_end;
    } else if (fmt == '3') {
        /* KHU3: packed 4-bit для всех 256 символов */
        if (input_size < 137) return 0;

        uint8_t base = input[8]; /* base = min_code_len - 1 */

        for (int i = 0; i < 256; i += 2) {
            uint8_t packed = input[9 + i / 2];
            uint8_t hi = (packed >> 4) & 0x0F;
            uint8_t lo = packed & 0x0F;
            code_lengths[i]     = (hi > 0) ? (uint8_t)(base + hi) : 0;
            code_lengths[i + 1] = (lo > 0) ? (uint8_t)(base + lo) : 0;
        }
        data_offset = 137;
    } else {
        /* KHUF: классический формат */
        if (input_size < 264) return 0;
        memcpy(code_lengths, &input[8], 256);
        data_offset = 264;
    }

    /* --- Canonical Huffman reconstruction --- */
    HuffmanContext ctx;
    huff_init(&ctx);

    int symbols_by_len[256];
    int sym_count = 0;
    for (int i = 0; i < 256; ++i) {
        if (code_lengths[i] > 0)
            symbols_by_len[sym_count++] = i;
    }

    /* Stable sort by (length, symbol) */
    for (int i = 0; i < sym_count - 1; ++i) {
        for (int j = i + 1; j < sym_count; ++j) {
            if (code_lengths[symbols_by_len[j]] < code_lengths[symbols_by_len[i]] ||
                (code_lengths[symbols_by_len[j]] == code_lengths[symbols_by_len[i]] &&
                 symbols_by_len[j] < symbols_by_len[i])) {
                int tmp = symbols_by_len[i];
                symbols_by_len[i] = symbols_by_len[j];
                symbols_by_len[j] = tmp;
            }
        }
    }

    /* Назначаем канонические коды */
    uint32_t code = 0;
    uint8_t prev_len = 0;
    for (int i = 0; i < sym_count; ++i) {
        int sym = symbols_by_len[i];
        uint8_t len = code_lengths[sym];
        if (prev_len > 0)
            code = (code + 1) << (len - prev_len);
        ctx.codes[sym].code = code;
        ctx.codes[sym].length = len;
        prev_len = len;
    }

    /* Строим дерево из кодов */
    HuffNode root_node;
    memset(&root_node, 0, sizeof(root_node));
    root_node.symbol = -1;
    ctx.tree_root = &root_node;

    for (int i = 0; i < sym_count; ++i) {
        int sym = symbols_by_len[i];
        uint32_t c = ctx.codes[sym].code;
        uint8_t len = ctx.codes[sym].length;
        HuffNode *node = ctx.tree_root;

        for (int b = len - 1; b >= 0; --b) {
            int bit = (c >> b) & 1;
            if (bit == 0) {
                if (!node->left) {
                    node->left = huff_alloc_node(&ctx);
                    if (!node->left) return 0;
                }
                node = node->left;
            } else {
                if (!node->right) {
                    node->right = huff_alloc_node(&ctx);
                    if (!node->right) return 0;
                }
                node = node->right;
            }
        }
        node->symbol = sym;
    }

    /* --- Декодирование битового потока --- */
    size_t out_pos = 0;
    HuffNode *current = ctx.tree_root;
    int bit_pos = 7;
    size_t byte_pos = data_offset;

    while (out_pos < orig_size && byte_pos < input_size) {
        int bit = (input[byte_pos] >> bit_pos) & 1;
        bit_pos--;
        if (bit_pos < 0) {
            bit_pos = 7;
            byte_pos++;
        }

        current = bit ? current->right : current->left;
        if (!current) break;

        if (current->symbol >= 0) {
            output[out_pos++] = (uint8_t)current->symbol;
            current = ctx.tree_root;
        }
    }

    return out_pos;
}


/* ============================================================================
 * tANS — Table-based Asymmetric Numeral Systems
 *
 * Быстрее Huffman, сжатие близко к arithmetic coding.
 * Использует одну lookup-таблицу для encode/decode.
 * ============================================================================ */

#define ANS_TABLE_LOG  10
#define ANS_TABLE_SIZE (1 << ANS_TABLE_LOG)  /* 1024 */
#define ANS_LOWER      ANS_TABLE_SIZE
#define ANS_UPPER      (ANS_TABLE_SIZE * 2)

typedef struct {
    uint16_t new_state;
    uint8_t  symbol;
    uint8_t  nb_bits;    /* Сколько бит вывести */
} ANSDecodeEntry;

typedef struct {
    uint16_t start;
    uint16_t freq;
} ANSSymbolInfo;

typedef struct {
    ANSDecodeEntry decode_table[ANS_TABLE_SIZE];
    ANSSymbolInfo  symbol_info[HUFFMAN_MAX_SYMBOLS];
    uint64_t       freq[HUFFMAN_MAX_SYMBOLS];
    int            nsymbols;
} ANSContext;

static void ans_init(ANSContext *ctx) {
    memset(ctx, 0, sizeof(*ctx));
}

static int ans_build_table(ANSContext *ctx) {
    /* Нормализация частот до ANS_TABLE_SIZE */
    uint64_t total = 0;
    for (int i = 0; i < 256; ++i) total += ctx->freq[i];
    if (total == 0) return -1;

    /* Выделяем нормализованные частоты (мин 1 для каждого ненулевого) */
    uint16_t norm_freq[256] = {0};
    uint32_t assigned = 0;
    ctx->nsymbols = 0;

    for (int i = 0; i < 256; ++i) {
        if (ctx->freq[i] > 0) {
            norm_freq[i] = (uint16_t)((ctx->freq[i] * ANS_TABLE_SIZE + total / 2) / total);
            if (norm_freq[i] == 0) norm_freq[i] = 1;
            assigned += norm_freq[i];
            ctx->nsymbols++;
        }
    }

    /* Корректировка до точного ANS_TABLE_SIZE */
    while (assigned > ANS_TABLE_SIZE) {
        /* Уменьшаем самый частый */
        int max_sym = -1;
        uint16_t max_freq = 0;
        for (int i = 0; i < 256; ++i) {
            if (norm_freq[i] > max_freq) {
                max_freq = norm_freq[i];
                max_sym = i;
            }
        }
        if (max_sym < 0 || norm_freq[max_sym] <= 1) break;
        norm_freq[max_sym]--;
        assigned--;
    }
    while (assigned < ANS_TABLE_SIZE) {
        int max_sym = -1;
        uint16_t max_freq = 0;
        for (int i = 0; i < 256; ++i) {
            if (norm_freq[i] > max_freq) {
                max_freq = norm_freq[i];
                max_sym = i;
            }
        }
        if (max_sym < 0) break;
        norm_freq[max_sym]++;
        assigned++;
    }

    /* Заполнение symbol_info */
    uint16_t cumulative = 0;
    for (int i = 0; i < 256; ++i) {
        ctx->symbol_info[i].start = cumulative;
        ctx->symbol_info[i].freq = norm_freq[i];
        cumulative += norm_freq[i];
    }

    /* Построение decode table */
    uint16_t next_state[256];
    for (int i = 0; i < 256; ++i) {
        next_state[i] = norm_freq[i];
    }

    for (int x = 0; x < ANS_TABLE_SIZE; ++x) {
        /* Определяем символ для позиции x */
        uint16_t pos = 0;
        int sym = 0;
        for (int i = 0; i < 256; ++i) {
            if (norm_freq[i] > 0) {
                if (x >= pos && x < pos + norm_freq[i]) {
                    sym = i;
                    break;
                }
                pos += norm_freq[i];
            }
        }

        ctx->decode_table[x].symbol = (uint8_t)sym;
        uint16_t f = norm_freq[sym];
        if (f == 0) f = 1;

        /* nb_bits = table_log - floor(log2(f)) */
        int log2f = 0;
        uint16_t tmp = f;
        while (tmp > 1) { tmp >>= 1; log2f++; }
        ctx->decode_table[x].nb_bits = (uint8_t)(ANS_TABLE_LOG - log2f);
        ctx->decode_table[x].new_state = (uint16_t)(
            (f << ctx->decode_table[x].nb_bits) - ANS_TABLE_SIZE + (x - ctx->symbol_info[sym].start)
        );
    }

    return 0;
}

/*
 * ans_compress — сжатие через tANS
 *
 * Формат:
 *   [4 bytes] magic "KANS"
 *   [4 bytes] original_size (LE)
 *   [4 bytes] final_state (LE)
 *   [256 * 2 bytes] normalized frequencies
 *   [... bytes] bitstream (reverse order)
 */
size_t ans_compress(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_capacity
) {
    if (!input || !output || input_size == 0) return 0;

    ANSContext ctx;
    ans_init(&ctx);

    /* Подсчёт частот */
    for (size_t i = 0; i < input_size; ++i) {
        ctx.freq[input[i]]++;
    }

    if (ans_build_table(&ctx) != 0) return 0;

    /* Кодирование (в обратном порядке — ANS requirement) */
    /* Сначала кодируем в промежуточный буфер битов */
    size_t max_bits = input_size * 12 + 256;  /* Верхняя граница */
    uint8_t *bit_buf = (uint8_t *)malloc(max_bits);
    if (!bit_buf) return 0;
    size_t bit_pos = 0;

    uint32_t state = ANS_LOWER;

    /* Кодируем задом наперёд */
    for (size_t i = input_size; i > 0; --i) {
        uint8_t sym = input[i - 1];
        uint16_t f = ctx.symbol_info[sym].freq;
        if (f == 0) { free(bit_buf); return 0; }

        /* Renormalize: выводим биты пока state слишком большой */
        while (state >= f * ANS_UPPER / ANS_TABLE_SIZE) {
            if (bit_pos / 8 >= max_bits) { free(bit_buf); return 0; }
            /* Выводим младший бит state */
            size_t byte_idx = bit_pos / 8;
            int bit_idx = bit_pos % 8;
            if (bit_idx == 0) bit_buf[byte_idx] = 0;
            bit_buf[byte_idx] |= ((state & 1) << bit_idx);
            bit_pos++;
            state >>= 1;
        }

        /* Encode symbol */
        state = ((state / f) * ANS_TABLE_SIZE) + (state % f) + ctx.symbol_info[sym].start;
    }

    /* Заголовок */
    size_t pos = 0;
    size_t header_size = 4 + 4 + 4 + 4 + 256 * 2; /* +4 для num_bits */
    size_t data_bytes = (bit_pos + 7) / 8;
    if (output_capacity < header_size + data_bytes) {
        free(bit_buf);
        return 0;
    }

    /* Magic */
    output[pos++] = 'K'; output[pos++] = 'A';
    output[pos++] = 'N'; output[pos++] = 'S';

    /* Original size */
    output[pos++] = (uint8_t)(input_size & 0xFF);
    output[pos++] = (uint8_t)((input_size >> 8) & 0xFF);
    output[pos++] = (uint8_t)((input_size >> 16) & 0xFF);
    output[pos++] = (uint8_t)((input_size >> 24) & 0xFF);

    /* Final state */
    output[pos++] = (uint8_t)(state & 0xFF);
    output[pos++] = (uint8_t)((state >> 8) & 0xFF);
    output[pos++] = (uint8_t)((state >> 16) & 0xFF);
    output[pos++] = (uint8_t)((state >> 24) & 0xFF);

    /* Num bits (NEW field) */
    output[pos++] = (uint8_t)(bit_pos & 0xFF);
    output[pos++] = (uint8_t)((bit_pos >> 8) & 0xFF);
    output[pos++] = (uint8_t)((bit_pos >> 16) & 0xFF);
    output[pos++] = (uint8_t)((bit_pos >> 24) & 0xFF);

    /* Normalized frequencies */
    for (int i = 0; i < 256; ++i) {
        output[pos++] = (uint8_t)(ctx.symbol_info[i].freq & 0xFF);
        output[pos++] = (uint8_t)((ctx.symbol_info[i].freq >> 8) & 0xFF);
    }

    /* Bitstream (уже в обратном порядке) */
    memcpy(&output[pos], bit_buf, data_bytes);
    pos += data_bytes;

    free(bit_buf);
    return pos;
}

/*
 * ans_decompress — декомпрессия tANS
 */
size_t ans_decompress(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_capacity
) {
    if (!input || !output) return 0;

    size_t header_size = 4 + 4 + 4 + 4 + 256 * 2; /* Updated header size */
    if (input_size < header_size) return 0;

    /* Проверка magic */
    if (input[0] != 'K' || input[1] != 'A' || input[2] != 'N' || input[3] != 'S') {
        return 0;
    }

    uint32_t orig_size = (uint32_t)input[4] | ((uint32_t)input[5] << 8)
                       | ((uint32_t)input[6] << 16) | ((uint32_t)input[7] << 24);
    uint32_t final_state = (uint32_t)input[8] | ((uint32_t)input[9] << 8)
                         | ((uint32_t)input[10] << 16) | ((uint32_t)input[11] << 24);
    uint32_t num_bits = (uint32_t)input[12] | ((uint32_t)input[13] << 8)
                      | ((uint32_t)input[14] << 16) | ((uint32_t)input[15] << 24);

    if (orig_size > output_capacity) return 0;

    /* Восстановление таблицы */
    ANSContext ctx;
    ans_init(&ctx);

    uint16_t cumulative = 0;
    for (int i = 0; i < 256; ++i) {
        size_t off = 16 + i * 2; /* Offset shifted by 4 */
        uint16_t f = (uint16_t)input[off] | ((uint16_t)input[off + 1] << 8);
        ctx.symbol_info[i].freq = f;
        ctx.symbol_info[i].start = cumulative;
        cumulative += f;
        if (f > 0) ctx.freq[i] = f;
    }

    if (ans_build_table(&ctx) != 0) return 0;

    /* Декодирование */
    const uint8_t *data = &input[header_size];
    size_t data_size = input_size - header_size;
    /* У нас теперь точное количество бит! */
    size_t bit_pos = (size_t)num_bits;

    uint32_t state = final_state;

    for (size_t i = 0; i < orig_size; ++i) {
        /* Decode symbol */
        uint16_t x = (uint16_t)(state % ANS_TABLE_SIZE);
        ANSDecodeEntry *entry = &ctx.decode_table[x];
        output[i] = entry->symbol;

        /* Update state */
        uint16_t f = ctx.symbol_info[entry->symbol].freq;
        if (f == 0) break;
        state = f * (state / ANS_TABLE_SIZE) + (state % ANS_TABLE_SIZE) - ctx.symbol_info[entry->symbol].start;

        /* Read bits to renormalize */
        while (state < ANS_LOWER && bit_pos > 0) {
            bit_pos--;
            size_t byte_idx = bit_pos / 8;
            int bit_idx = bit_pos % 8;
            if (byte_idx < data_size) {
                int bit = (data[byte_idx] >> bit_idx) & 1;
                state = (state << 1) | bit;
            }
        }
    }

    return orig_size;
}


/* ============================================================================
 * Публичное API — выбор метода
 * ============================================================================ */

typedef enum {
    KOLIBRI_ENTROPY_HUFFMAN = 0,
    KOLIBRI_ENTROPY_ANS     = 1,
    KOLIBRI_ENTROPY_AUTO    = 2,  /* Выбирает лучший */
} KolibriCompressMethod;

/*
 * kolibri_entropy_compress — unified API
 */
size_t kolibri_entropy_compress(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_capacity,
    KolibriCompressMethod method
) {
    if (method == KOLIBRI_ENTROPY_AUTO) {
        /* Пробуем оба, выбираем меньший */
        uint8_t *tmp = (uint8_t *)malloc(output_capacity);
        if (!tmp) return huffman_compress(input, input_size, output, output_capacity);

        size_t huff_size = huffman_compress(input, input_size, output, output_capacity);
        size_t ans_size = ans_compress(input, input_size, tmp, output_capacity);

        if (ans_size > 0 && (huff_size == 0 || ans_size < huff_size)) {
            memcpy(output, tmp, ans_size);
            free(tmp);
            return ans_size;
        }
        free(tmp);
        return huff_size;
    }

    if (method == KOLIBRI_ENTROPY_HUFFMAN) {
        return huffman_compress(input, input_size, output, output_capacity);
    }
    return ans_compress(input, input_size, output, output_capacity);
}

size_t kolibri_entropy_decompress(
    const uint8_t *input, size_t input_size,
    uint8_t *output, size_t output_capacity
) {
    if (input_size < 4) return 0;

    /* Определяем метод по magic */
    if (input[0] == 'K' && input[1] == 'H' && input[2] == 'U' &&
        (input[3] == 'F' || input[3] == '2' || input[3] == '3')) {
        return huffman_decompress(input, input_size, output, output_capacity);
    }
    if (input[0] == 'K' && input[1] == 'A' && input[2] == 'N' &&
        (input[3] == 'S' || input[3] == '2')) {
        return ans_decompress(input, input_size, output, output_capacity);
    }
    return 0;
}
