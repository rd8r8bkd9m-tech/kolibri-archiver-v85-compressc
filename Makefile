CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Iinclude -I/opt/homebrew/include
LDFLAGS ?= -L/opt/homebrew/lib -ldivsufsort -lm -lpthread

BIN_DIR := bin
SRC_DIR := src
APP_DIR := apps

TARGET := $(BIN_DIR)/kolibri_archiver

SRCS := \
	$(APP_DIR)/kolibri_archiver.c \
	$(SRC_DIR)/compress.c \
	$(SRC_DIR)/huffman_ans.c \
	$(SRC_DIR)/random.c

.PHONY: all clean bench

all: $(TARGET)

$(TARGET): $(SRCS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(SRCS) -o $@ $(LDFLAGS)

bench: $(TARGET)
	./scripts/bench_compressc.sh src/compress.c

clean:
	rm -rf $(BIN_DIR) docs/benchmark_compressc_latest.csv
