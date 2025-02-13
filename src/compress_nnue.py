#!/usr/bin/env python3

import hashlib
import os
from pprint import pprint
import struct
import subprocess
import sys

import numpy as np


use_8bit = True
zero_unused_weights = True

# HIDDEN_WIDTH = 96
with open("nnue.h", "r") as f:
    for row in f:
        if row.startswith("#define HIDDEN_WIDTH"):
            HIDDEN_WIDTH = int(row.split()[-1])

COLORS = [0, 1]  # white, black
PIECE_TYPES = [1, 2, 3, 4, 5, 6]  # pawn, knight, bishop, rook, queen, king
SQUARES = range(64)

# their perspective starts at this index
THEIR_PERSPECTIVE_START = len(PIECE_TYPES) * len(SQUARES) * HIDDEN_WIDTH


def read_16bit_number(infile):
    return struct.unpack('h', infile.read(2))[0]

def write_16bit_number(outfile, value):
    outfile.write(struct.pack('h', value))

def write_8bit_number(outfile, value):
    # value = min(max(value, -128), 127)
    if value < -128:
        print(f"value < -128: {value}")
    elif value > 127:
        print(f"value > 127: {value}")
    outfile.write(struct.pack('b', value))

# flips the board vertically for non-stm (ie. a2 -> a7)
def relative_square(pov, sq):
    return sq ^ (pov * 56)

# 0 = white pawn, 1 = white knight, ...
# 8 = black pawn, 9 = black knight, ...
def make_piece(c, pt):
    return (c << 3) + pt

def encode_sleb128(value):
    # Encodes an integer to signed LEB128 format
    result = bytearray()
    more = True
    while more:
        byte = value & 0x7F
        value >>= 7
        if (value == 0 and (byte & 0x40) == 0) or (value == -1 and (byte & 0x40) != 0):
            more = False
        else:
            byte |= 0x80
        result.append(byte)
    return result

# 16-bit params from .nnue output files from bullet trainer (quantised.nnue)
def load_nnue(input_file):
    ftW, ftB, oW, oB = [], [], [], None
    with open(input_file, "rb") as infile:
        # feature transformer weights
        for color in COLORS:
            for pt in PIECE_TYPES:
                for sq in SQUARES:
                    for _ in range(HIDDEN_WIDTH):
                        value = read_16bit_number(infile)
                        ftW.append(value)
        # feature transformer biases
        for _ in range(HIDDEN_WIDTH):
            ftB.append(read_16bit_number(infile))
        # output weights - two perspectives: stm, not-stm
        for _ in range(2 * HIDDEN_WIDTH):
            oW.append(read_16bit_number(infile))
        # output bias
        oB = read_16bit_number(infile)
    return (ftW, ftB, oW, oB)


def zero_unused(ftW):
    for pt in PIECE_TYPES:
        for pov in COLORS:
            for i in range(HIDDEN_WIDTH):
                for sq in SQUARES:
                    idx = pov * THEIR_PERSPECTIVE_START + (pt-1) * len(SQUARES) * HIDDEN_WIDTH + sq * HIDDEN_WIDTH + i
                    if pt == 1 and (sq < 8 or sq > 55):
                        # zero all weights for pawns on 1st and 8th ranks
                        ftW[idx] = 0
                    elif pov == 0 and pt == 6 and (sq % 8) > 3:
                        # zero all king weights for our perspetive on EFGH files (horizontal mirroring)
                        ftW[idx] = 0


def map_extremes_ftW(pt_color_square_weights):
    unique_weights = sorted(set([int(v) for v in pt_color_square_weights]))
    unused_8bit_weights = set()
    for i in range(-128, 128):
        if i not in unique_weights:
            unused_8bit_weights.add(i)
    unused_8bit_weights = sorted(unused_8bit_weights)
    # extremes = [int(v) for v in sorted(pt_color_square_weights, key=lambda x: -abs(x))[:n_extremes]]
    extremes = sorted(set([int(v) for v in pt_color_square_weights if v < -128 or v > 127]), key=lambda x: abs(x))
    print("# unique weights outside 8-bit range:", len(extremes))
    print("extremes:", extremes)

    extreme_to_8bit = {}
    unused_8bit_to_extreme = {}

    negative_extremes = sorted(w for w in extremes if w < 0)
    positive_extremes = sorted([w for w in extremes if w > 0], key=lambda x: -x)

    print("unused 8-bit -> 16-bit mappings:")
    for i in range(len(negative_extremes)):
        weight = unused_8bit_weights.pop(0)
        print(f"  {weight} -> {negative_extremes[i]}")
        extreme_to_8bit[negative_extremes[i]] = weight
        unused_8bit_to_extreme[weight] = negative_extremes[i]

    for i in range(len(positive_extremes)):
        weight = unused_8bit_weights.pop()
        print(f"  {weight} -> {positive_extremes[i]}")
        extreme_to_8bit[positive_extremes[i]] = weight
        unused_8bit_to_extreme[weight] = positive_extremes[i]

    remapped_weights = []
    for weight in pt_color_square_weights:
        if extreme_to_8bit.get(weight):
            remapped_weights.append(extreme_to_8bit[weight])
        else:
            remapped_weights.append(weight)

    print(f"  # unique remapped weights: {len(set(remapped_weights))}")

    return remapped_weights, unused_8bit_to_extreme


def quantize_and_map_extremes_ftW(pt_color_square_weights, num_groups_of_four):
    """ quantizes extremes in groups of 4 to reduce # of unique weights
    """
    unique_weights = sorted(set([int(v) for v in pt_color_square_weights]))
    print(f"# unique weights: {len(unique_weights)}")
    unused_8bit_weights = set()
    for i in range(-128, 128):
        if i not in unique_weights:
            unused_8bit_weights.add(i)
    unused_8bit_weights = sorted(unused_8bit_weights, key=lambda x: abs(x))
    # extremes = [int(v) for v in sorted(pt_color_square_weights, key=lambda x: -abs(x))[:n_extremes]]
    extremes = sorted(set([int(v) for v in pt_color_square_weights if v < -128 or v > 127]))
    print("# unused 8bit weights:", len(unused_8bit_weights))
    print("# unique weights outside 8-bit range:", len(extremes))
    print("extremes:", extremes)

    extreme_to_8bit = {}
    unused_8bit_to_extreme = {}

    negative_extremes = sorted(w for w in extremes if w < 0)
    positive_extremes = sorted([w for w in extremes if w > 0], key=lambda x: -x)
    print(negative_extremes)
    print(positive_extremes)

    # quantize extremes in groups of 4
    new_pt_color_square_weights = pt_color_square_weights.copy()
    for i in range(num_groups_of_four):
        if len(negative_extremes) > len(positive_extremes):
            group_of_four = negative_extremes[:4]
            negative_extremes = negative_extremes[4:]
        else:
            group_of_four = positive_extremes[:4]
            positive_extremes = positive_extremes[4:]
        print("group of 4:", group_of_four)
        if group_of_four[0] < 0:
            quant_target = min(group_of_four)
            for j,weight in enumerate(new_pt_color_square_weights):
                if weight in group_of_four:
                    new_pt_color_square_weights[j] = quant_target
        pt_color_square_weights = new_pt_color_square_weights

    unique_weights = sorted(set([int(v) for v in pt_color_square_weights]))
    print(f"# unique weights after quantization: {len(unique_weights)}")

    extremes = sorted(set([int(v) for v in pt_color_square_weights if v < -128 or v > 127]))
    negative_extremes = sorted(w for w in extremes if w < 0)
    positive_extremes = sorted([w for w in extremes if w > 0], key=lambda x: -x)

    print("mapping previously-unused 8-bit -> 16-bit weights:")
    for i in range(len(negative_extremes)):
        weight = unused_8bit_weights.pop(0)
        print(f"  {weight} -> {negative_extremes[i]}")
        extreme_to_8bit[negative_extremes[i]] = weight
        unused_8bit_to_extreme[weight] = negative_extremes[i]

    for i in range(len(positive_extremes)):
        weight = unused_8bit_weights.pop()
        print(f"  {weight} -> {positive_extremes[i]}")
        extreme_to_8bit[positive_extremes[i]] = weight
        unused_8bit_to_extreme[weight] = positive_extremes[i]

    remapped_weights = []
    for weight in pt_color_square_weights:
        if extreme_to_8bit.get(weight):
            remapped_weights.append(extreme_to_8bit[weight])
        else:
            remapped_weights.append(weight)

    print(f"  # unique remapped weights: {len(set(remapped_weights))}")

    return remapped_weights, unused_8bit_to_extreme


# piece -> color -> square -> embedding vector
def rearrange_ftW(ftW):
    reordered_ftW = []
    for pt in PIECE_TYPES:
        for pov in COLORS:
            pov_pt_weights = []
            for sq in SQUARES:
                pov_pt_sq_weights = []
                for i in range(HIDDEN_WIDTH):
                    idx = pov * THEIR_PERSPECTIVE_START + (pt-1) * len(SQUARES) * HIDDEN_WIDTH + sq * HIDDEN_WIDTH + i
                    pov_pt_sq_weights.append(ftW[idx])
                pov_pt_weights.append(pov_pt_sq_weights)
            pt_color_square_weights = np.array(pov_pt_weights).T.flatten()

            mn = pt_color_square_weights.min()
            mx = pt_color_square_weights.max()
            offset = round(abs(abs(mx) - abs(mn)) / 2)
            print(f"pt: {pt}, pov: {pov} - range: [{mn}, {mx}], offset: {offset}, offset range: [{mn+offset}, {mx+offset}]")

            unique_weights = sorted(set([int(v) for v in pt_color_square_weights]))
            print("  # unique weights:", len(unique_weights))
            print("  unique weights:", unique_weights)

            unused_8bit_weights = set()
            for i in range(-128, 128):
                if i not in unique_weights:
                    unused_8bit_weights.add(i)
            print("  # unused 8-bit weights:", len(unused_8bit_weights))
            print("  unused 8-bit weights:", sorted(unused_8bit_weights))
            print()
            reordered_ftW.append(pt_color_square_weights)

    print("remapping our pawn weights:")
    remapped_our_pawn_ftW, our_pawn_mappings = map_extremes_ftW(reordered_ftW[0])
    print()
    print("remapping their pawn weights:")
    remapped_their_pawn_ftW, their_pawn_mappings = map_extremes_ftW(reordered_ftW[1])
    print()

    print("remapping our knight weights:")
    remapped_our_knight_ftW, our_knight_mappings = map_extremes_ftW(reordered_ftW[2])
    print()
    print("remapping their knight weights:")
    remapped_their_knight_ftW, their_knight_mappings = map_extremes_ftW(reordered_ftW[3])
    print()

    print("remapping our bishop weights:")
    remapped_our_bishop_ftW, our_bishop_mappings = map_extremes_ftW(reordered_ftW[4])
    print()
    print("remapping their bishop weights:")
    remapped_their_bishop_ftW, their_bishop_mappings = map_extremes_ftW(reordered_ftW[5])
    print()

    print("remapping our rook weights:")
    remapped_our_rook_ftW, our_rook_mappings = map_extremes_ftW(reordered_ftW[6])
    print()
    print("remapping their rook weights:")
    remapped_their_rook_ftW, their_rook_mappings = map_extremes_ftW(reordered_ftW[7])
    print()

    # queen weights don't fit within an 8-bit range
    print("remapping our queen weights:")
    remapped_our_queen_ftW, our_queen_mappings = map_extremes_ftW(reordered_ftW[8])
    print()
    print("remapping their queen weights:")
    remapped_their_queen_ftW, their_queen_mappings = map_extremes_ftW(reordered_ftW[9])
    print()

    print("remapping our king weights:")
    remapped_our_king_ftW, our_king_mappings = map_extremes_ftW(reordered_ftW[10])
    print()
    print("remapping their king weights:")
    remapped_their_king_ftW, their_king_mappings = map_extremes_ftW(reordered_ftW[11])
    print()

    print("remapped their king weights:")
    print([int(v) for v in remapped_their_king_ftW])

    reordered_ftW_new = [
        # reordered_ftW[0], reordered_ftW[1],
        remapped_our_pawn_ftW, remapped_their_pawn_ftW,

        # reordered_ftW[2], reordered_ftW[3],
        remapped_our_knight_ftW, remapped_their_knight_ftW,

        # reordered_ftW[4], reordered_ftW[5],
        remapped_our_bishop_ftW, remapped_their_bishop_ftW,

        # reordered_ftW[6], reordered_ftW[7],
        remapped_our_rook_ftW, remapped_their_rook_ftW,

        # reordered_ftW[8], reordered_ftW[9],
        remapped_our_queen_ftW, remapped_their_queen_ftW,

        # reordered_ftW[10], reordered_ftW[11],
        remapped_our_king_ftW, remapped_their_king_ftW,
    ]


    reordered_ftW_new = [
        remapped_our_pawn_ftW, remapped_their_pawn_ftW,
        remapped_our_knight_ftW, remapped_their_knight_ftW,
        remapped_our_bishop_ftW, remapped_their_bishop_ftW,
        remapped_our_rook_ftW, remapped_their_rook_ftW,
        remapped_our_queen_ftW, remapped_their_queen_ftW,
        remapped_our_king_ftW, remapped_their_king_ftW,
    ]


    print("REMAPPED OUR KING")
    print([int(v) for v in remapped_our_king_ftW])
    print()
    print()
    print("REMAPPED THEIR KING")
    print([int(v) for v in remapped_their_king_ftW])
    print()
    print()

    def mapping_to_c_code(piece_type, pov, mappings):
        # return ", ".join([f"[{unused_8bit+128}] = {mapped_16bit}" for unused_8bit,mapped_16bit in mappings.items()])
        rows = []
        for unused_8bit, mapped_16bit in mappings.items():
            rows.append(f"        m[{piece_type}][{pov}][{unused_8bit + 128}] = {mapped_16bit};")
        return "\n".join(rows).strip()

    print(f"""
    int16_t map_unused_8bit_to_actual_16bit_weight(int piece_type, int pov, int8_t value) {{
        static const int16_t mappings[6][2][256] = {0};

        // our pawn
        {mapping_to_c_code(0, 0, our_pawn_mappings)}
        // their pawn
        {mapping_to_c_code(0, 1, their_pawn_mappings)}

        // our knight
        {mapping_to_c_code(1, 0, our_knight_mappings)}
        // their knight
        {mapping_to_c_code(1, 1, their_knight_mappings)}

        // our bishop
        {mapping_to_c_code(2, 0, our_bishop_mappings)}
        // their bishop
        {mapping_to_c_code(2, 1, their_bishop_mappings)}

        // our rook
        {mapping_to_c_code(3, 0, our_rook_mappings)}
        // their rook
        {mapping_to_c_code(3, 1, their_rook_mappings)}

        // our queen
        {mapping_to_c_code(4, 0, our_queen_mappings)}
        // their queen
        {mapping_to_c_code(4, 1, their_queen_mappings)}

        // our king
        {mapping_to_c_code(5, 0, our_king_mappings)}
        // their king
        {mapping_to_c_code(5, 1, their_king_mappings)}
      }};
    }};
    """)

    flattened_ftW = []
    for piece_pov_weights in reordered_ftW_new:
        for weight in piece_pov_weights:
            flattened_ftW.append(int(weight))
    
    return flattened_ftW


def write_nnue(output_file, nnue_params, use_8bit=False, use_leb128=False):
    ftW, ftB, oW, oB = nnue_params
    with open(output_file, "wb") as outfile:
        # write feature transformer weights
        for value in ftW:
            if use_8bit:
                write_8bit_number(outfile, value)
            elif use_leb128:
                outfile.write(encode_sleb128(value))
            else:
                write_16bit_number(outfile, value)

        # write feature transformer biases
        for i in range(HIDDEN_WIDTH):
            if use_8bit:
                # print(f"8-bit: ftB[{i}] = {ftB[i]}")
                write_8bit_number(outfile, ftB[i])
            elif use_leb128:
                outfile.write(encode_sleb128(ftB[i]))
            else:
                write_16bit_number(outfile, ftB[i])

        # write output weights
        for i in range(2 * HIDDEN_WIDTH):
            if use_leb128:
                outfile.write(encode_sleb128(oW[i]))
            else:
                write_16bit_number(outfile, oW[i])

        # write output bias
        write_16bit_number(outfile, oB)


def nnue_filename():
    return "HL64-qa101-qb160-S2-T77novT79maraprmay.nnue"
    with open("nnue.h", "r") as f:
        for row in f:
            if row.startswith("#define EvalFile"):
                return row.split("#define EvalFile ")[-1].strip().strip('"')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        input_file = nnue_filename()
    else:
        input_file = sys.argv[1]

    print(f"input: {input_file}")
    print(f"hidden width: {HIDDEN_WIDTH}")
    print(f"8-bit: {use_8bit}, zero unused: {zero_unused_weights}")

    # load the original weights, then rearrange them before writing to output file
    (ftW, ftB, oW, oB) = load_nnue(input_file)
    ftW_hash = hashlib.sha256(str(sorted(ftW)).encode()).hexdigest()[:32]

    if zero_unused_weights:
        zero_unused(ftW)

    # rearrange and quantize weights to fit into 8 bits
    new_ftW = rearrange_ftW(ftW)
    new_ftW_hash = hashlib.sha256(str(sorted(new_ftW)).encode()).hexdigest()[:32]

    # print(ftW)
    print(f"oB = {oB}\n")
    print(f"before: {ftW_hash}")
    print(f"# unique weights: {len(set(ftW)):>6}")
    print(f"# zeroes:         {len([w for w in ftW if w == 0]):>6}")
    print(f"sum weights:      {sum(ftW):>6}")
    print(f"# ftW:            {len(ftW):>6}\n")

    # print(new_ftW)
    print(f"after: {new_ftW_hash}")
    # print(f"# unique weights: {len(set(new_ftW))}")
    print(f"# zeroes:         {len([w for w in new_ftW if w == 0]):>6}")
    print(f"sum weights:      {sum(new_ftW):>6}")
    print(f"# ftW:            {len(new_ftW):>6}\n")

    output_file_16bit = input_file.replace(".nnue", ".rearranged.16-bit.nnue")
    write_nnue(output_file_16bit, (new_ftW, ftB, oW, oB), use_8bit=False)

    output_file_16bit_leb128 = input_file.replace(".nnue", ".rearranged.16-bit.leb128.nnue")
    write_nnue(output_file_16bit_leb128, (new_ftW, ftB, oW, oB), use_leb128=True)

    output_file_8bit = input_file.replace(".nnue", ".rearranged.8-bit.nnue")
    write_nnue(output_file_8bit, (new_ftW, ftB, oW, oB), use_8bit=True)

    # print sizes of input file and variations
    size = os.path.getsize(input_file)
    print(f'{size:>5} bytes - input: {input_file}')

    size = os.path.getsize(output_file_16bit)
    print(f'{size:>5} bytes - output 16-bit: {output_file_16bit}')

    size = os.path.getsize(output_file_16bit_leb128)
    print(f'{size:>5} bytes - output 16-bit leb128: {output_file_16bit_leb128}')

    size = os.path.getsize(output_file_8bit)
    print(f'{size:>5} bytes - output 8-bit: {output_file_8bit}')

    def compress_file(filename):
        compressed_filename = f"{filename}.xz"
        subprocess.run(['xz', '-9', '-k', '-f', filename], check=True)
        size = os.path.getsize(compressed_filename)
        print(f'{size:>5} bytes - compressed: {compressed_filename}')

    print()
    compress_file(input_file)
    compress_file(output_file_16bit)
    compress_file(output_file_16bit_leb128)
    compress_file(output_file_8bit)

    # compressed_output_file = f"{output_file}.xz"
    # subprocess.run(['xz', '-9', '-k', '-f', output_file], check=True)
    # size = os.path.getsize(compressed_output_file)
    # print(f'{size:>5} bytes - compressed output: {compressed_output_file}\n')
    # print(f"output: {output_file}")
