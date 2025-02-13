/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2016 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include "misc.h"
#include "thread.h"

// Version number. If Version is left empty, then compile date in the format
// DD-MM-YY and show in engine_info.
char Version[] = "";

pthread_mutex_t ioMutex = PTHREAD_MUTEX_INITIALIZER;

// print engine_info() prints the full name of the current Stockfish version.
// This will be either "Stockfish <Tag> DD-MM-YY" (where DD-MM-YY is the
// date when the program was compiled) or "Stockfish <Version>", depending
// on whether Version is empty.

#ifndef MINIMAL
void print_engine_info(void)
{
  printf("17");
  printf(
#ifdef USE_AVX512
         " AVX512"
#elif USE_PEXT
         " BMI2"
#elif USE_AVX2
         " AVX2"
#elif USE_NEON
         " NEON"
#elif USE_POPCNT
         " POPCNT"
#endif
#ifdef USE_VNNI
         "-VNNI"
#endif
         "\n");
  fflush(stdout);
}
#endif

// xorshift64star Pseudo-Random Number Generator
// This class is based on original code written and dedicated
// to the public domain by Sebastiano Vigna (2014).
// It has the following characteristics:
//
//  -  Outputs 64-bit numbers
//  -  Passes Dieharder and SmallCrush test batteries
//  -  Does not require warm-up, no zeroland to escape
//  -  Internal state is a single 64-bit integer
//  -  Period is 2^64 - 1
//  -  Speed: 1.60 ns/call (Core i7 @3.40GHz)
//
// For further analysis see
//   <http://vigna.di.unimi.it/ftp/papers/xorshift.pdf>

void prng_init(PRNG *rng, uint64_t seed)
{
  rng->s = seed;
}

uint64_t prng_rand(PRNG *rng)
{
  uint64_t s = rng->s;

  s ^= s >> 12;
  s ^= s << 25;
  s ^= s >> 27;
  rng->s = s;

  return s * 2685821657736338717LL;
}

uint64_t prng_sparse_rand(PRNG *rng)
{
  uint64_t r1 = prng_rand(rng);
  uint64_t r2 = prng_rand(rng);
  uint64_t r3 = prng_rand(rng);
  return r1 & r2 & r3;
}

ssize_t getline(char **lineptr, size_t *n, FILE *stream)
{
  if (*n == 0)
    *lineptr = malloc(*n = 100);

  int c = 0;
  size_t i = 0;
  while ((c = getc(stream)) != EOF) {
    (*lineptr)[i++] = c;
    if (i == *n)
      *lineptr = realloc(*lineptr, *n += 100);
    if (c == '\n') break;
  }
  (*lineptr)[i] = 0;
  return i;
}

FD open_file(const char *name)
{
  return open(name, O_RDONLY);
}

void close_file(FD fd)
{
  close(fd);
}

size_t file_size(FD fd)
{
  struct stat statbuf;
  fstat(fd, &statbuf);
  return statbuf.st_size;
}

const void *map_file(FD fd, map_t *map)
{
  *map = file_size(fd);
  void *data = mmap(NULL, *map, PROT_READ, MAP_SHARED, fd, 0);
#ifdef MADV_RANDOM
  madvise(data, *map, MADV_RANDOM);
#endif
  return data == MAP_FAILED ? NULL : data;
}

void unmap_file(const void *data, map_t map)
{
  if (!data) return;
  munmap((void *)data, map);
}

void *allocate_memory(size_t size, alloc_t *alloc)
{
  void *ptr = NULL;

  size_t alignment = 1;
  size_t allocSize = size + alignment - 1;

#if defined(__APPLE__) && defined(VM_FLAGS_SUPERPAGE_SIZE_2MB)
  ptr = mmap(NULL, allocSize, PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

#else
  ptr = mmap(NULL, allocSize, PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

  alloc->ptr = ptr;
  alloc->size = allocSize;
  return (void *)(((uintptr_t)ptr + alignment - 1) & ~(alignment - 1));

#endif
}

void free_memory(alloc_t *alloc)
{
  munmap(alloc->ptr, alloc->size);
}
