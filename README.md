# Kolibri Archiver V85 (compress.c)

Отдельный проект архиватора, выделенный из Calibri AI, на базе `backend/src/compress.c` (ветка v85).

## Что внутри
- `src/compress.c` — основной движок сжатия/распаковки.
- `apps/kolibri_archiver.c` — CLI (`compress`, `decompress`, `bench` и др.).
- Минимальные зависимости движка: `src/huffman_ans.c`, `src/random.c`.

## Сборка
```bash
make all
```

## Быстрый тест
```bash
./bin/kolibri_archiver test src/compress.c
```

## Бенчмарк на compress.c
```bash
./scripts/bench_compressc.sh src/compress.c
```

Результат сохраняется в:
- `docs/benchmark_compressc_latest.csv`
