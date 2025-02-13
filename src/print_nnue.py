#!/usr/bin/env python3

from pprint import pprint
import struct
import sys

from rearrange_nnue import (
    nnue_filename,
    HIDDEN_WIDTH, COLORS, PIECE_TYPES, SQUARES
)


def piece_type_name(pt):
    if pt == 1: return "Pawn"
    elif pt == 2: return "Knight"
    elif pt == 3: return "Bishop"
    elif pt == 4: return "Rook"
    elif pt == 5: return "Queen"
    elif pt == 6: return "King"


def process_file(input_file):
    num_16_bit_numbers_read = 0
    num_ftW_8_bit_compat = 0
    num_ftW_7_bit_compat = 0
    num_ftW_6_bit_compat = 0
    num_ftW_4_bit_compat = 0
    num_ftW_3_bit_compat = 0
    num_ftW_2_bit_compat = 0
    num_ftW_1_bit_compat = 0

    num_ftW_row_16bit = 0
    num_ftW_row_8bit = 0
    num_ftW_row_7bit = 0
    num_ftW_row_6bit = 0
    num_ftW_row_5bit = 0
    num_ftW_row_4bit = 0
    num_ftW_row_3bit = 0
    num_ftW_row_2bit = 0
    num_ftW_row_1bit = 0

    ftW_weights_count = {}
    ftW = []
    ftB = []
    oW = []

    with open(input_file, 'rb') as infile:
        # feature transformer weights
        for color in COLORS:
            for pt in PIECE_TYPES:
                pt_weights = []
                for sq in SQUARES:
                    row = []
                    for _ in range(HIDDEN_WIDTH):
                        # Read 2 bytes (16 bits)
                        data = infile.read(2)
                        num_16_bit_numbers_read += 1

                        # Unpack as signed 16-bit integer
                        value = struct.unpack('h', data)[0]
                        row.append(value)
                        ftW.append(value)
                        pt_weights.append(value)

                        if not ftW_weights_count.get(value):
                            ftW_weights_count[value] = 0
                        ftW_weights_count[value] += 1

                        if value >= -128 and value <= 127:
                            num_ftW_8_bit_compat += 1
                        if value >= -64 and value <= 63:
                            num_ftW_7_bit_compat += 1
                        if value >= -32 and value <= 31:
                            num_ftW_6_bit_compat += 1
                        if value >= -8 and value <= 7:
                            num_ftW_4_bit_compat += 1
                        if value >= -4 and value <= 3:
                            num_ftW_3_bit_compat += 1
                        if value >= -2 and value <= 1:
                            num_ftW_2_bit_compat += 1
                        if value >= 0 and value <= 1:
                            num_ftW_1_bit_compat += 1

                    if all(v >= 0 and v <= 1 for v in row):
                        num_ftW_row_1bit += 1
                    if all(v >= -2 and v <= 1 for v in row):
                        num_ftW_row_2bit += 1
                    if all(v >= -4 and v <= 3 for v in row):
                        num_ftW_row_3bit += 1
                    elif all(v >= -8 and v <= 7 for v in row):
                        num_ftW_row_4bit += 1
                    elif all(v >= -16 and v <= 15 for v in row):
                        num_ftW_row_5bit += 1
                    elif all(v >= -32 and v <= 31 for v in row):
                        num_ftW_row_6bit += 1
                    elif all(v >= -64 and v <= 63 for v in row):
                        num_ftW_row_7bit += 1
                    elif all(v >= -128 and v <= 127 for v in row):
                        num_ftW_row_8bit += 1
                    else:
                        num_ftW_row_16bit += 1
                       
                    # print(f"ftW[{color}][{pt}][{sq}] = {{{", ".join(f'{v:>5}' for v in row)}}};")
                    print(f"ftW[{color}][{pt-1}][{sq}] = {{{", ".join(str(v) for v in row)}}};")
                perspective = "Our" if color == 0 else "Their"
                pt_st = piece_type_name(pt)
                print(perspective, pt_st)
                print(f"  range: {min(pt_weights)} -> {max(pt_weights)}")
                print(f"  # unique: {len(set(pt_weights))}")
                print(f"  # unique outside [-16, 16]: {len(set([w for w in pt_weights if w < -16 or w > 16]))}")
                print(f"  unique weights: {sorted(set(pt_weights))}")
                print()

        # feature transformer biases
        print(f"FT biases: {HIDDEN_WIDTH}")
        for _ in range(HIDDEN_WIDTH):
            data = infile.read(2)
            value = struct.unpack('h', data)[0]
            ftB.append(value)
        print(f"ftB = {{{", ".join(str(v) for v in ftB)}}};")
        print()

        # output weights 
        print(f"output weights: {HIDDEN_WIDTH * 2} = {HIDDEN_WIDTH} x 2")
        for _ in range(2 * HIDDEN_WIDTH):
            data = infile.read(2)
            value = struct.unpack('h', data)[0]
            oW.append(value)
        print(f"oW = {{{", ".join(str(v) for v in oW)}}};")
        print()

        # output bias
        print("output bias:")
        data = infile.read(2)
        value = struct.unpack('h', data)[0]
        print(f"oB = {value};")
        print()

    if False:
        pprint(sorted(ftW_weights_count.items(), key=lambda x: abs(10_000 * x[1]) - abs(x[0])))

    print("400+: ", sorted([w for w in ftW if w >= 400]))
    print("300+: ", sorted([w for w in ftW if w > 300 and w < 400]))
    print()
    print(f"Feature transformer weights (HL {HIDDEN_WIDTH}) that fit in:")
    print(f"# 16-bit:  {num_16_bit_numbers_read:5}")
    print(f"# 8-bit:   {num_ftW_8_bit_compat:5} ({100*num_ftW_8_bit_compat/num_16_bit_numbers_read:.3}%)")
    print(f"# 7-bit:   {num_ftW_7_bit_compat:5} ({100*num_ftW_7_bit_compat/num_16_bit_numbers_read:.3}%)")
    print(f"# 6-bit:   {num_ftW_6_bit_compat:5} ({100*num_ftW_6_bit_compat/num_16_bit_numbers_read:.3}%)")
    print(f"# 4-bit:   {num_ftW_4_bit_compat:5} ({100*num_ftW_4_bit_compat/num_16_bit_numbers_read:.3}%)")
    print(f"# 3-bit:   {num_ftW_3_bit_compat:5} ({100*num_ftW_3_bit_compat/num_16_bit_numbers_read:.3}%)")
    print(f"# 2-bit:   {num_ftW_2_bit_compat:5} ({100*num_ftW_2_bit_compat/num_16_bit_numbers_read:.3}%)")
    print(f"# 1-bit:   {num_ftW_1_bit_compat:5} ({100*num_ftW_1_bit_compat/num_16_bit_numbers_read:.3}%)")
    print()
    print(f"Rows counted:  {num_ftW_row_4bit + num_ftW_row_6bit + num_ftW_row_7bit + num_ftW_row_8bit + num_ftW_row_16bit + num_ftW_row_5bit + num_ftW_row_3bit}")
    print(f"# rows 16-bit: {num_ftW_row_16bit}")
    print(f"# rows  8-bit: {num_ftW_row_8bit}")
    print(f"# rows  7-bit: {num_ftW_row_7bit}")
    print(f"# rows  6-bit: {num_ftW_row_6bit}")
    print(f"# rows  5-bit: {num_ftW_row_5bit}")
    print(f"# rows  4-bit: {num_ftW_row_4bit}")
    print(f"# rows  3-bit: {num_ftW_row_3bit}")
    print(f"# rows  2-bit: {num_ftW_row_2bit}")
    print(f"# rows  1-bit: {num_ftW_row_1bit}")
    print()
    print(f"# unique weights: {len(ftW_weights_count.items())}")
    print(f"# zeroes: ", len([w for w in ftW if w == 0]))
    print(f"ftW range: [{min(ftW)}, {max(ftW)}]")
    print(f"ftB range: [{min(ftB)}, {max(ftB)}]")
    print(f"oW range:  [{min(oW)}, {max(oW)}]")
    return (ftW, ftB, oW, ftW_weights_count)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        input_file = "HL64-qa101-qb160-S2-T77novT79maraprmay.nnue"
    else:
        input_file = sys.argv[1]
    process_file(input_file)
    print()
    print(input_file)
