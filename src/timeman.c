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

#include <float.h>
#include <math.h>

#include "search.h"
#include "timeman.h"
#include "uci.h"

struct TimeManagement Time; // Our global time management struct

// tm_init() is called at the beginning of the search and calculates the
// time bounds allowed for the current game ply. We currently support:
// 1) x basetime (+z increment)
// 2) x moves in y seconds (+z increment)

void time_init(Color us, int ply)
{
  // UCI Move Overhead
  const int moveOverhead    = 10;

  // Maximum move horizon of 50 moves
  int mtg = 50;

  // if less than one second, gradually reduce mtg
  if (Limits.time[us] < 1000 && ((double)mtg / Limits.time[us] > 0.05))
  {
    mtg = Limits.time[us] * 0.05;
  }

  // optScale is a percentage of available time to use for the current move.
  // maxScale is a multiplier applied to optimumTime.
  double optScale, maxScale;

  Time.startTime = Limits.startTime;

  // Make sure that timeLeft > 0 since we may use it as a divisor
  TimePoint timeLeft = max(1, Limits.time[us] + Limits.inc[us] * (mtg - 1) - moveOverhead * (2 + mtg));

  // Use extra time with larger increments
  double optExtra = clamp(1.0 + 12.0 * Limits.inc[us] / Limits.time[us], 1.0, 1.12);

  // x basetime (+z increment)
  // If there is a healthy increment, timeLeft can exceed actual available
  // game time for the current move, so also cap to 20% of available game time.
  optScale = min(0.0120 + pow(ply + 3.0, 0.45) * 0.0039,
                  0.2 * Limits.time[us] / (double)timeLeft)
             * optExtra;
  maxScale = min(7.0, 4.0 + ply / 12.0);

  // Never use more than 80% of the available time for this move
  Time.optimumTime = optScale * timeLeft;
  Time.maximumTime = min(0.8 * Limits.time[us] - moveOverhead, maxScale * Time.optimumTime);

#ifdef MINIMAL
  Time.optimumTime += Time.optimumTime / 3;
#else
  if (option_value(OPT_PONDER))
    Time.optimumTime += Time.optimumTime / 3;
#endif
}
