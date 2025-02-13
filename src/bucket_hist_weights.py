#!/usr/bin/env python3

import sys

import numpy as np
import matplotlib.pyplot as plt

from rearrange_nnue import (
    nnue_filename, load_nnue, zero_unused,
    HIDDEN_WIDTH, COLORS, PIECE_TYPES, SQUARES
)

if len(sys.argv) != 2:
    input_file = nnue_filename()
else:
    input_file = sys.argv[1]

(ftW, ftB, oW, oB) = load_nnue(input_file)
zero_unused(ftW)

i = 0
ft_weights = []
for c in COLORS:
    pov_weights = []
    for pt in PIECE_TYPES:
        pov_pt_weights = []
        for sq in SQUARES:
            pov_pt_sq_weights = ftW[i:i+HIDDEN_WIDTH]
            print(pov_pt_sq_weights)
            pov_pt_weights.append(pov_pt_sq_weights)
            i += HIDDEN_WIDTH
        pov_weights.append(pov_pt_weights.copy())
    ft_weights.append(pov_weights)

weights1 = ft_weights[0]   #      stm, np.random.rand(6, 64, HL)
weights2 = ft_weights[1]   #  not-stm, np.random.rand(6, 64, HL)
w_min = np.percentile(ftW, 10)
w_max = np.percentile(ftW, 90)

if "/" in input_file:
    title = input_file.split("/")[1]
else:
    title = input_file

# Visualization
plt.figure(figsize=(12, 10))  # Adjust size to fit all six blocks
plt.suptitle(title)
for i in range(6):
    # Plot first set
    plt.subplot(6, 2, 2 * i + 1)
    plt.hist(weights1[i], bins=50, edgecolor='k')
    
    # Plot second set
    plt.subplot(6, 2, 2 * i + 2, anchor='W')
    plt.hist(weights2[i], bins=50, edgecolor='k')

plt.tight_layout()
plt.show()
