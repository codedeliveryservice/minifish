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

#ifndef MOVEPICK_H
#define MOVEPICK_H

#include <string.h>   // For memset

#include "movegen.h"
#include "position.h"
#include "search.h"
#include "types.h"

#define stats_clear(s) memset(s, 0, sizeof(*s))

#define PAWN_HISTORY_SIZE 512

#define PIECE_TO_HISTORY_GRAIN 128
#define PAWN_CORR_HIST_GRAIN 8

inline int pawn_structure(Position *pos) {
  return pawn_key() & (PAWN_HISTORY_SIZE - 1);
}

// continuation history
inline int ch_pawn_structure(Position *pos) {
  return pawn_key() & (16384 - 1);
}

inline int non_pawn_index(Position *pos, Color c) {
  return non_pawn_key(c) & (8192 - 1);
}

static const int CounterMovePruneThreshold = 0;

INLINE void cms_update(PieceToHistory cms, Piece pc, Square to, int v)
{
  // cms[pc][to] += v - cms[pc][to] * abs(v) / 16384;
  // return;

  int16_t scaledValue = PIECE_TO_HISTORY_GRAIN * cms[pc][to];
  // cms[pc][to] = (scaledValue + v - scaledValue * abs(v) / 30000) / PIECE_TO_HISTORY_GRAIN;
  // return;

  cms[pc][to] = clamp(
    (scaledValue + v - scaledValue * abs(v) / 50000) / PIECE_TO_HISTORY_GRAIN,
    INT8_MIN,
    INT8_MAX
  );
}

// static eval correction history
INLINE void correction_history_update(CorrectionHistory ch, Color c, Position *pos, int v)
{
  int16_t scaledValue = PAWN_CORR_HIST_GRAIN * ch[c][ch_pawn_structure(pos)];
  ch[c][ch_pawn_structure(pos)] = clamp(
    (scaledValue + v - scaledValue * abs(v) / 1024) / PAWN_CORR_HIST_GRAIN,
    INT8_MIN,
    INT8_MAX
  );
}

// non-pawn correction history
INLINE void non_pawn_correction_history_update(NonPawnCorrectionHistory npch, Color c, Color stm, Position *pos, int v)
{
  npch[c][stm][non_pawn_index(pos, c)] += v - npch[c][stm][non_pawn_index(pos, c)] * abs(v) / 1024;
}

INLINE void history_update(ButterflyHistory history, Color c, Move m, int v)
{
  m &= 4095;
  history[c][m] = clamp(
    (64 * history[c][m] + v - 64 * history[c][m] * abs(v) / 7183) / 64,
    INT8_MIN,
    INT8_MAX
  );
}

INLINE void cpth_update(CapturePieceToHistory history, Piece pc, Square to,
                        int captured, int v)
{
  history[pc][to][captured] += v - history[pc][to][captured] * abs(v) / 10692;
}

enum {
  ST_MAIN_SEARCH, ST_CAPTURES_INIT, ST_GOOD_CAPTURES, ST_KILLERS, ST_KILLERS_2,
  ST_QUIET_INIT, ST_QUIET, ST_BAD_CAPTURES,

  ST_EVASION, ST_EVASIONS_INIT, ST_ALL_EVASIONS,

  ST_QSEARCH, ST_QCAPTURES_INIT, ST_QCAPTURES, ST_QCHECKS,

  ST_PROBCUT, ST_PROBCUT_INIT, ST_PROBCUT_2
};

Move next_move(const Position *pos, bool skipQuiets);

// Initialisation of move picker data.

INLINE void mp_init(const Position *pos, Move ttm, Depth d, int ply)
{
  assert(d > 0);

  Stack *st = pos->st;

  st->depth = d;
  st->mp_ply = ply;

  Square prevSq = to_sq((st-1)->currentMove);
  st->countermove = (*pos->counterMoves)[piece_on(prevSq)][prevSq];
  st->mpKillers[0] = st->killers[0];
  st->mpKillers[1] = st->killers[1];

  st->ttMove = ttm;
  st->stage = checkers() ? ST_EVASION : ST_MAIN_SEARCH;
  if (!ttm || !is_pseudo_legal(pos, ttm))
    st->stage++;
}

INLINE void mp_init_q(const Position *pos, Move ttm, Depth d, Square s)
{
  assert(d <= 0);

  Stack *st = pos->st;

  st->ttMove = ttm;
  st->stage = checkers() ? ST_EVASION : ST_QSEARCH;
  if (!(   ttm
        && is_pseudo_legal(pos, ttm)))
    st->stage++;

  st->depth = d;
  st->recaptureSquare = s;
}

INLINE void mp_init_pc(const Position *pos, Move ttm, Value th)
{
  assert(!checkers());

  Stack *st = pos->st;

  st->threshold = th;

  st->ttMove = ttm;
  st->stage = ST_PROBCUT;

  // In ProbCut we generate captures with SEE higher than the given
  // threshold.
  if (!(ttm && is_pseudo_legal(pos, ttm) && is_capture(pos, ttm)
            && see_test(pos, ttm, th)))
    st->stage++;
}

#endif
