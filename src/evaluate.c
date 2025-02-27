/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2018 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad

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

#include <assert.h>
#include <stdio.h>

#include "bitboard.h"
#include "evaluate.h"
#include "position.h"

Value evaluate(const Position *pos)
{
  Value nnue = nnue_evaluate(&pos->st->accumulator, stm());

  int material = 592 * (piece_count(WHITE, PAWN) + piece_count(BLACK, PAWN)) + non_pawn_material();
  Value v      = (nnue * (77045 + material) + pos->optimism[stm()] * (7446 + material)) / 88903;

  // Damp down the evalation linearly when shuffling
  v -= v * rule50_count() / 256;

  return clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
}
