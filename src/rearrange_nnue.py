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
    value = min(max(value, -128), 127)
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


# piece -> color -> square -> embedding vector
def rearrange_ftW(ftW):
    reordered_ftW = []
    for i in range(HIDDEN_WIDTH):
        for pov in COLORS:
            for pt in PIECE_TYPES:
                for sq in SQUARES:
                    idx = pov * THEIR_PERSPECTIVE_START + (pt-1) * len(SQUARES) * HIDDEN_WIDTH + sq * HIDDEN_WIDTH + i
                    reordered_ftW.append(ftW[idx])
    return reordered_ftW


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
            pov_pt_weights = np.array(pov_pt_weights)
            reordered_ftW.append(pov_pt_weights.T.flatten())
    return [int(v) for v in np.array(reordered_ftW).flatten()]


# for each piece, then each square, put their weights next to ours
def rearrange_ftW_ours_then_theirs(ftW):
    reordered_ftW = []
    for pt in PIECE_TYPES:
        our_pt_sq_weights = []
        their_pt_sq_weights = []
        for sq in SQUARES:
            # our perspective - piece on a particular square
            our_piece_sq_idx = (pt-1) * len(SQUARES) * HIDDEN_WIDTH + sq * HIDDEN_WIDTH
            our_pt_sq_weights.append(ftW[our_piece_sq_idx:our_piece_sq_idx + HIDDEN_WIDTH])

            # their perspective - piece on a particular square
            their_piece_sq_idx = THEIR_PERSPECTIVE_START + our_piece_sq_idx
            their_pt_sq_weights.append(ftW[their_piece_sq_idx:their_piece_sq_idx + HIDDEN_WIDTH])

        reordered_ftW.extend(np.array(our_pt_sq_weights).T.flatten().tolist())
        reordered_ftW.extend(np.array(their_pt_sq_weights).T.flatten().tolist())
    return reordered_ftW


def rearrange_ftW_sq_vec(ftW):
    reordered_ftW = []
    square_vectors = []  # HL x 12 = 576
    for pt in PIECE_TYPES:
        for pov in COLORS:
            for i in range(HIDDEN_WIDTH):
                sq_vector = []
                for sq in SQUARES:
                    idx = pov * THEIR_PERSPECTIVE_START + (pt-1) * len(SQUARES) * HIDDEN_WIDTH + sq * HIDDEN_WIDTH + i
                    sq_vector.append(ftW[idx])
                square_vectors.append(sq_vector)
    square_vectors = sorted(square_vectors, key=lambda v: sum(v))
    return np.array(square_vectors).flatten()


def write_nnue(output_file, nnue_params, use_8bit=False, use_leb128=False):
    ftW, ftB, oW, oB = nnue_params
    with open(output_file, "wb") as outfile:
        for value in ftW:
            if use_8bit:
                write_8bit_number(outfile, value)
            elif use_leb128:
                outfile.write(encode_sleb128(value))
            else:
                write_16bit_number(outfile, value)
        for i in range(HIDDEN_WIDTH):
            if use_8bit:
                write_8bit_number(outfile, ftB[i])
            elif use_leb128:
                outfile.write(encode_sleb128(ftB[i]))
            else:
                write_16bit_number(outfile, ftB[i])
        for i in range(2 * HIDDEN_WIDTH):
            if use_8bit:
                write_8bit_number(outfile, oW[i])
            elif use_leb128:
                outfile.write(encode_sleb128(oW[i]))
            else:
                write_16bit_number(outfile, oW[i])
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
    # print(ftW)
    print(f"oB = {oB}\n")
    print(f"before: {ftW_hash}")
    print(f"# unique weights: {len(set(ftW)):>6}")
    print(f"# zeroes:         {len([w for w in ftW if w == 0]):>6}")
    print(f"sum weights:      {sum(ftW):>6}")
    print(f"# ftW:            {len(ftW):>6}\n")

    if zero_unused_weights:
        zero_unused(ftW)

    new_ftW = rearrange_ftW(ftW)
    new_ftW_hash = hashlib.sha256(str(sorted(new_ftW)).encode()).hexdigest()[:32]
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
