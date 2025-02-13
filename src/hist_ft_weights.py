#!/usr/bin/env python3

from pprint import pprint
import struct
import sys

import numpy as np
import matplotlib.pyplot as plt

from rearrange_nnue import HIDDEN_WIDTH, nnue_filename


colors = [0, 1]  # white, black
piece_types = [0, 1, 2, 3, 4, 5]
squares = range(64)

# bullet footer
ignore_values = set([30050, 27756, 29797])


def process_file(input_file):
    num_16_bit_numbers_read = 0
    num_ftW_8_bit_compat = 0
    num_ftW_7_bit_compat = 0
    num_ftW_6_bit_compat = 0
    num_ftW_4_bit_compat = 0

    ftW_weights_count = {}
    ftW = []

    with open(input_file, 'rb') as infile:
        # feature transformer weights
        for color in colors:
            for pt in piece_types:
                for sq in squares:
                    row = []
                    for _ in range(HIDDEN_WIDTH):
                        # Read 2 bytes (16 bits)
                        data = infile.read(2)
                        num_16_bit_numbers_read += 1

                        # Unpack as signed 16-bit integer
                        value = struct.unpack('h', data)[0]
                        row.append(str(value))
                        ftW.append(value)

    num_buckets = len(set(ftW))
    data = np.array(ftW)

    # plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=num_buckets, edgecolor='k')
    plt.title(f'{input_file}: {len(ftW)} ({num_buckets} unique weights)')
    plt.xlabel('weight')
    plt.ylabel('frequency')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        input_file = nnue_filename()
    else:
        input_file = sys.argv[1]
    print(input_file)
    process_file(input_file)
