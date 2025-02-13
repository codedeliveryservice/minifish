/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2020 The Stockfish developers

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
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "evaluate.h"
#include "misc.h"
#include "movegen.h"
#include "movepick.h"
#include "search.h"
#include "settings.h"
#include "timeman.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"

#define load_rlx(x) atomic_load_explicit(&(x), memory_order_relaxed)
#define store_rlx(x,y) atomic_store_explicit(&(x), y, memory_order_relaxed)

LimitsType Limits;

// Different node types, used as template parameter
enum { NonPV, PV };

INLINE int futility_margin(Depth d, bool improving, bool opponentWorsening) {
  int futilityMult = 180;
  return futilityMult * d - futilityMult * improving / 2 - futilityMult * opponentWorsening / 3;
}

// Reductions lookup tables, initialized at startup
static int Reductions[MAX_MOVES]; // [depth or moveNumber]

INLINE Depth reduction(int i, Depth d, int mn, Value delta, Value rootDelta)
{
  int reductionScale = Reductions[d] * Reductions[mn];
  return (reductionScale + 1239 - delta * 795 / rootDelta) + (!i && reductionScale > 1341) * 1135;
}

INLINE int futility_move_count(bool improving, Depth depth)
{
//  return (3 + depth * depth) / (2 - improving);
  return improving ? 3 + depth * depth : (3 + depth * depth) / 2;
}

static int correction_value(Position *pos)
{
  Color us = stm();
  int pcv = PAWN_CORR_HIST_GRAIN * (*pos->correctionHistory)[us][ch_pawn_structure(pos)];
  int wnpcv = (*pos->nonPawnCorrectionHistory)[WHITE][us][non_pawn_index(pos, WHITE)];
  int bnpcv = (*pos->nonPawnCorrectionHistory)[BLACK][us][non_pawn_index(pos, BLACK)];

  return (10 * pcv + 8 * (wnpcv + bnpcv)) / 128;
}

// Add correctionHistory value to raw staticEval and guarantee evaluation
// does not hit the tablebase range.
static Value to_corrected_static_eval(Value v, const int cv) {
  return clamp(v + cv, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
}


// History and stats update bonus, based on depth
static Value stat_bonus(Depth depth)
{
  int d = depth;
  return min(388 * d - 421, 2121);
}

static Value stat_malus(Depth depth)
{
  int d = depth;
  return min(416 * d - 284, 1580);
}

// Add a small random component to draw evaluations to keep search dynamic
// and to avoid three-fold blindness. (Yucks, ugly hack)
static Value value_draw(Position *pos)
{
  return VALUE_DRAW + 2 * (pos->nodes & 1) - 1;
}

static Value search_PV(Position *pos, Stack *ss, Value alpha, Value beta,
    Depth depth);
static Value search_NonPV(Position *pos, Stack *ss, Value alpha, Depth depth,
    bool cutNode);

static Value qsearch_PV_true(Position *pos, Stack *ss, Value alpha, Value beta,
    Depth depth);
static Value qsearch_PV_false(Position *pos, Stack *ss, Value alpha, Value beta,
    Depth depth);
static Value qsearch_NonPV_true(Position *pos, Stack *ss, Value alpha,
    Depth depth);
static Value qsearch_NonPV_false(Position *pos, Stack *ss, Value alpha,
    Depth depth);

static Value value_to_tt(Value v, int ply);
static Value value_from_tt(Value v, int ply, int r50c);
static void update_pv(Move *pv, Move move, Move *childPv);
static void update_continuation_histories(Stack *ss, Piece pc, Square s, int bonus);
static void update_quiet_histories(const Position *pos, Stack *ss, Move move, int bonus);
static void check_time(void);
static void stable_sort(RootMove *rm, int num);
static void uci_print_pv(Position *pos, Depth depth, Value alpha, Value beta);
static int extract_ponder_from_tt(RootMove *rm, Position *pos);

// search_init() is called during startup to initialize various lookup tables

void search_init(void)
{
  for (int i = 1; i < MAX_MOVES; i++)
    Reductions[i] = 21.14 * log(i);
}


// search_clear() resets search state to zero, to obtain reproducible results

void search_clear(void)
{
  if (!settings.ttSize) {
    delayedSettings.clear = true;
    return;
  }

  Time.availableNodes = 0;

  tt_clear();
  Position *pos = Threads.pos;

  stats_clear(pos->counterMoveHistory);
  for (int j = 0; j < 16; j++)
    for (int k = 0; k < 64; k++)
      (*pos->counterMoveHistory)[0][0][j][k] = CounterMovePruneThreshold - 1;

  stats_clear(pos->counterMoves);
  stats_clear(pos->mainHistory);
  stats_clear(pos->captureHistory);

  mainThread.previousScore = VALUE_INFINITE;
  mainThread.bestPreviousAverageScore = VALUE_INFINITE;
  mainThread.previousTimeReduction = 1;
}


// mainthread_search() is called by the main thread when the program
// receives the UCI 'go' command. It searches from the root position and
// outputs the "bestmove".

void mainthread_search(void)
{
  Position *pos = Threads.pos;
  Color us = stm();
  time_init(us, game_ply());
  tt_new_search();
  char buf[16];

#ifdef NNUE
  switch (useNNUE) {
  case EVAL_HYBRID:
    printf("info string Hybrid NNUE evaluation using %s enabled.\n", option_string_value(OPT_EVAL_FILE));
    break;
  case EVAL_PURE:
    printf("info string Pure NNUE evaluation using %s enabled.\n", option_string_value(OPT_EVAL_FILE));
    break;
  case EVAL_CLASSICAL:
    printf("info string Classical evaluation enabled.\n");
    break;
  }
#endif

  if (pos->rootMoves->size > 0) {
    Threads.pos->bestMoveChanges = 0;
    thread_search(pos); // Let's start searching!
  }

  // When we reach the maximum depth, we can arrive here without Threads.stop
  // having been raised. However, if we are pondering or in an infinite
  // search, the UCI protocol states that we shouldn't print the best
  // move before the GUI sends a "stop" or "ponderhit" command. We
  // therefore simply wait here until the GUI sends one of those commands
  // (which also raises Threads.stop).
  LOCK(Threads.lock);
  if (!Threads.stop && Threads.ponder) {
    Threads.sleeping = true;
    UNLOCK(Threads.lock);
    thread_wait(pos, &Threads.stop);
  } else
    UNLOCK(Threads.lock);

  // Stop the other threads if they have not stopped already
  Threads.stop = true;

  // Wait until all threads have finished
  if (pos->rootMoves->size > 0) {
    // do nothing
  } else {
    pos->rootMoves->move[0].pv[0] = 0;
    pos->rootMoves->move[0].pvSize = 1;
    pos->rootMoves->size++;
    printf("info depth 0 score %s\n",
           uci_value(buf, checkers() ? -VALUE_MATE : VALUE_DRAW));
    fflush(stdout);
  }

  mainThread.previousScore = pos->rootMoves->move[0].score;
  mainThread.bestPreviousAverageScore = pos->rootMoves->move[0].averageScore;

  flockfile(stdout);
  printf("bestmove %s", uci_move(buf, pos->rootMoves->move[0].pv[0], is_chess960()));

  if (pos->rootMoves->move[0].pvSize > 1 || extract_ponder_from_tt(&pos->rootMoves->move[0], pos))
    printf(" ponder %s", uci_move(buf, pos->rootMoves->move[0].pv[1], is_chess960()));

  printf("\n");
  fflush(stdout);
  funlockfile(stdout);
}


// thread_search() is the main iterative deepening loop. It calls search()
// repeatedly with increasing depth until the allocated thinking time has
// been consumed, the user stops the search, or the maximum search depth is
// reached.

void thread_search(Position *pos)
{
  Value bestValue, alpha, beta, delta;
  Move pv[MAX_PLY + 1];
  Move lastBestMove = 0;
  Depth lastBestMoveDepth = 0;
  double timeReduction = 1.0, totBestMoveChanges = 0;
  int iterIdx = 0;

  Stack *ss = pos->st; // At least the seventh element of the allocated array.
  for (int i = -7; i < 3; i++) {
    memset(SStackBegin(ss[i]), 0, SStackSize);
#ifdef NNUE
    ss[i].accumulator.state[WHITE] = ACC_INIT;
    ss[i].accumulator.state[BLACK] = ACC_INIT;
#endif
  }
  (ss-1)->endMoves = pos->moveList;

  for (int i = -7; i < 0; i++)
  {
    ss[i].history = &(*pos->counterMoveHistory)[0][0]; // Use as sentinel
    ss[i].staticEval = VALUE_NONE;
    ss[i].checkersBB = 0;
  }

  for (int i = 0; i <= MAX_PLY; i++)
    ss[i].ply = i;
  ss->pv = pv;

  bestValue = delta = alpha = -VALUE_INFINITE;
  beta = VALUE_INFINITE;
  pos->completedDepth = 0;

  if (mainThread.previousScore == VALUE_INFINITE)
    for (int i = 0; i < 4; i++)
      mainThread.iterValue[i] = VALUE_ZERO;
  else
    for (int i = 0; i < 4; i++)
      mainThread.iterValue[i] = mainThread.previousScore;

  RootMoves *rm = pos->rootMoves;
  int searchAgainCounter = 0;

  // Iterative deepening loop until requested to stop or the target depth
  // is reached.
  while (   ++pos->rootDepth < MAX_PLY
         && !Threads.stop
         && !(   Limits.depth
              && pos->rootDepth > Limits.depth))
  {
    // Age out PV variability metric
    totBestMoveChanges /= 2;

    // Save the last iteration's scores before first PV line is searched and
    // all the move scores except the (new) PV are set to -VALUE_INFINITE.
    for (int idx = 0; idx < rm->size; idx++)
      rm->move[idx].previousScore = rm->move[idx].score;

    if (!Threads.increaseDepth)
      searchAgainCounter++;

    // Former MultiPV loop. We perform a full root search for the PV line
    int pvFirst = 0, pvLast = rm->size;
    pos->pvLast = pvLast;
    pos->selDepth = 0;

    // Reset aspiration window starting size
    Value previousScore = rm->move[0].averageScore;
    delta = 19 + previousScore * previousScore / 16023;
    alpha = max(previousScore - delta, -VALUE_INFINITE);
    beta  = min(previousScore + delta,  VALUE_INFINITE);

    // Adjust optimism based on root move's averageScore (~4 Elo)
    pos->optimism[stm()]  = 137 * previousScore / (abs(previousScore) + 185);
    pos->optimism[!stm()] = -pos->optimism[stm()];

    // Start with a small aspiration window and, in the case of a fail
    // high/low, re-search with a bigger window until we're not failing
    // high/low anymore.
    int failedHighCnt = 0;
    while (true) {
      // Adjust the effective depth searched, but ensuring at least one effective increment for every
      // four searchAgain steps (see issue #2717).
      Depth adjustedDepth = max(1, pos->rootDepth - failedHighCnt - 3 * (searchAgainCounter + 1) / 4);
      bestValue = search_PV(pos, ss, alpha, beta, adjustedDepth);

      // Bring the best move to the front. It is critical that sorting
      // is done with a stable algorithm because all the values but the
      // first and eventually the new best one are set to -VALUE_INFINITE
      // and we want to keep the same order for all the moves except the
      // new PV that goes to the front. Note that in case of MultiPV
      // search the already searched PV lines are preserved.
      stable_sort(&rm->move[0], pvLast);

      // If search has been stopped, we break immediately. Sorting and
      // writing PV back to TT is safe because RootMoves is still
      // valid, although it refers to the previous iteration.
      if (Threads.stop)
        break;

      // When failing high/low give some update (without cluttering
      // the UI) before a re-search.
      if (   (bestValue <= alpha || bestValue >= beta)
          && time_elapsed() > 3000)
        uci_print_pv(pos, pos->rootDepth, alpha, beta);

      // In case of failing low/high increase aspiration window and
      // re-search, otherwise exit the loop.
      if (bestValue <= alpha) {
        beta = (alpha + beta) / 2;
        alpha = max(bestValue - delta, -VALUE_INFINITE);

        failedHighCnt = 0;
        Threads.stopOnPonderhit = false;
      } else if (bestValue >= beta) {
        beta = min(bestValue + delta, VALUE_INFINITE);
        failedHighCnt++;
      } else
        break;

      delta += delta / 4 + 2;

      assert(alpha >= -VALUE_INFINITE && beta <= VALUE_INFINITE);
    }

    // Sort the PV lines searched so far and update the GUI
    stable_sort(&rm->move[pvFirst], - pvFirst + 1);

    // skip_search:
    uci_print_pv(pos, pos->rootDepth, alpha, beta);

    if (!Threads.stop)
      pos->completedDepth = pos->rootDepth;

    if (rm->move[0].pv[0] != lastBestMove) {
      lastBestMove = rm->move[0].pv[0];
      lastBestMoveDepth = pos->rootDepth;
    }

    // Use part of the gained time from a previous stable move for this move
    totBestMoveChanges += Threads.pos->bestMoveChanges;
    Threads.pos->bestMoveChanges = 0;

    // Do we have time for the next iteration? Can we stop searching now?
    if (    use_time_management()
        && !Threads.stop
        && !Threads.stopOnPonderhit)
    {
      double fallingEval = (62.398 + 12.169 * (mainThread.bestPreviousAverageScore - bestValue)
                               +  6.071 * (mainThread.iterValue[iterIdx] - bestValue)) / 616.60;
      fallingEval = clamp(fallingEval, 0.539, 1.577);

      // If the best move is stable over several iterations, reduce time
      // accordingly
      timeReduction = lastBestMoveDepth + 8 < pos->completedDepth ? 1.723 : 0.914;
      double reduction = (1.4586 + mainThread.previousTimeReduction) / (1.8897 * timeReduction);

      double bestMoveInstability = 1.0828 + 1.7293 * totBestMoveChanges;

      double totalTime = time_optimum() * fallingEval * reduction * bestMoveInstability;

      // In the case of a single legal move, cap total time to 500ms.
      if (rm->size == 1)
        totalTime = min(500.0, totalTime);

      // Stop the search if we have exceeded the totalTime
      if (time_elapsed() > totalTime) {
        // If we are allowed to ponder do not stop the search now but
        // keep pondering until the GUI sends "ponderhit" or "stop".
        if (Threads.ponder)
          Threads.stopOnPonderhit = true;
        else
          Threads.stop = true;
      }
      else if (   !Threads.ponder
               && time_elapsed() > totalTime * 0.4615)
        Threads.increaseDepth = false;
      else
        Threads.increaseDepth = true;
    }

    mainThread.iterValue[iterIdx] = bestValue;
    iterIdx = (iterIdx + 1) & 3;
  }

  mainThread.previousTimeReduction = timeReduction;
}

// search_node() is the main search function template for both PV
// and non-PV nodes
INLINE Value search_node(Position *pos, Stack *ss, Value alpha, Value beta,
    Depth depth, bool cutNode, const int NT)
{
  const bool PvNode = NT == PV;
  const bool rootNode = PvNode && ss->ply == 0;

  // Check if we have an upcoming move which draws by repetition, or if the
  // opponent had an alternative move earlier to this position.
  if (   pos->st->pliesFromNull >= 3
      && alpha < VALUE_DRAW
      && !rootNode
      && has_game_cycle(pos, ss->ply))
  {
    alpha = value_draw(pos);
    if (alpha >= beta)
      return alpha;
  }

  // Dive into quiescense search when the depth reaches zero
  if (depth <= 0)
    return  PvNode
          ?   checkers()
            ? qsearch_PV_true(pos, ss, alpha, beta, 0)
            : qsearch_PV_false(pos, ss, alpha, beta, 0)
          :   checkers()
            ? qsearch_NonPV_true(pos, ss, alpha, 0)
            : qsearch_NonPV_false(pos, ss, alpha, 0);

  assert(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
  assert(PvNode || (alpha == beta - 1));
  assert(0 < depth && depth < MAX_PLY);
  assert(!(PvNode && cutNode));

  Move pv[MAX_PLY+1], capturesSearched[32], quietsSearched[32];
  TTEntry *tte;
  Key posKey;
  Move ttMove, move, excludedMove, bestMove;
  Depth extension, newDepth;
  Value bestValue, value, ttValue, eval, maxValue, probCutBeta;
  bool givesCheck, improving, opponentWorsening;
  bool captureOrPromotion, inCheck, moveCountPruning;
  bool ttCapture;
  Piece movedPiece;
  int moveCount, captureCount, quietCount;

  // Step 1. Initialize node
  inCheck = checkers();
  moveCount = captureCount = quietCount = ss->moveCount = 0;
  bestValue = -VALUE_INFINITE;
  maxValue = VALUE_INFINITE;

  // Check for the available remaining time
  if (load_rlx(pos->resetCalls)) {
    store_rlx(pos->resetCalls, false);
    pos->callsCnt = 1024;
  }

  if (--pos->callsCnt <= 0) {
    store_rlx(Threads.pos->resetCalls, true);
    check_time();
  }

  // Used to send selDepth info to GUI
  if (PvNode && pos->selDepth < ss->ply)
    pos->selDepth = ss->ply;

  if (!rootNode) {
    // Step 2. Check for aborted search and immediate draw
    if (load_rlx(Threads.stop) || is_draw(pos) || ss->ply >= MAX_PLY)
      return  ss->ply >= MAX_PLY && !inCheck ? evaluate(pos)
                                             : value_draw(pos);

    // Step 3. Mate distance pruning. Even if we mate at the next move our
    // score would be at best mate_in(ss->ply+1), but if alpha is already
    // bigger because a shorter mate was found upward in the tree then
    // there is no need to search because we will never beat the current
    // alpha. Same logic but with reversed signs applies also in the
    // opposite condition of being mated instead of giving mate. In this
    // case return a fail-high score.
    if (PvNode) {
      alpha = max(mated_in(ss->ply), alpha);
      beta = min(mate_in(ss->ply+1), beta);
      if (alpha >= beta)
        return alpha;
    } else { // avoid assignment to beta (== alpha+1)
      if (alpha < mated_in(ss->ply))
        return mated_in(ss->ply);
      if (alpha >= mate_in(ss->ply+1))
        return alpha;
    }
  } else {
    pos->rootDelta = beta - alpha;
  }

  assert(0 <= ss->ply && ss->ply < MAX_PLY);

  (ss+1)->ttPv = false;
  (ss+1)->excludedMove = bestMove = 0;
  (ss+2)->killers[0] = (ss+2)->killers[1] = 0;
  (ss+2)->cutoffCnt = 0;
  ss->doubleExtensions = (ss-1)->doubleExtensions;
  Square prevSq = to_sq((ss-1)->currentMove);

  // Initialize statScore to zero for the grandchildren of the current
  // position. So the statScore is shared between all grandchildren and only
  // the first grandchild starts with startScore = 0. Later grandchildren
  // start with the last calculated statScore of the previous grandchild.
  // This influences the reduction rules in LMR which are based on the
  // statScore of the parent position.
  if (!rootNode)
    (ss+2)->statScore = 0;

  // Step 4. Transposition table lookup. We don't want the score of a
  // partial search to overwrite a previous full search TT value, so we
  // use a different position key in case of an excluded move.
  excludedMove = ss->excludedMove;
  posKey = !excludedMove ? key() : key() ^ make_key(excludedMove);
  tte = tt_probe(posKey, &ss->ttHit);
  ttValue = ss->ttHit ? value_from_tt(tte_value(tte), ss->ply, rule50_count()) : VALUE_NONE;
  ttMove =  rootNode ? pos->rootMoves->move[0].pv[0]
          : ss->ttHit    ? tte_move(tte) : 0;
  ttCapture = ttMove && is_capture_or_promotion(pos, ttMove);
  if (!excludedMove)
    ss->ttPv = PvNode || (ss->ttHit && tte_is_pv(tte));

  // At non-PV nodes we check for an early TT cutoff.
  if (  !PvNode && !excludedMove && tte_depth(tte) > depth - (ttValue <= beta)
      && ttValue != VALUE_NONE // Possible in case of TT access race or if !ttHit
      && (ttValue >= beta ? (tte_bound(tte) & BOUND_LOWER)
                          : (tte_bound(tte) & BOUND_UPPER)))
  {
    // If ttMove is quiet, update move sorting heuristics on TT hit.
    if (ttMove && ttValue >= beta)
    {
      // Bonus for a quiet ttMove that fails high (~2 Elo)
      if (!ttCapture)
        update_quiet_histories(pos, ss, ttMove, stat_bonus(depth) * 1124 / 1024);

      // Extra penalty for early quiet moves of the previous ply
      if (prevSq != SQ_NONE && (ss-1)->moveCount <= 2 && !captured_piece())
        update_continuation_histories(ss-1, piece_on(prevSq), prevSq, -stat_malus(depth) * 1840 / 1024);
    }

    // Partial workaround for the graph history interaction problem
    // For high rule50 counts don't produce transposition table cutoffs.
    if (rule50_count() < 90)
      return ttValue;
  }

  // Step 5. Tablebase probe (removed)

  Value unadjustedStaticEval = VALUE_NONE;
  int correctionValue = correction_value(pos);

  // Step 6. Static evaluation of the position
  if (inCheck)
  {
    // Skip early pruning when in check
    ss->staticEval = eval = (ss - 2)->staticEval;
    improving = false;
    goto moves_loop;
  }
  else if (ss->ttHit)
  {
    // Never assume anything about values stored in TT
    if ((eval = tte_eval(tte)) == VALUE_NONE)
      eval = evaluate(pos);

    unadjustedStaticEval = ss->staticEval = eval;

    ss->staticEval = eval = to_corrected_static_eval(unadjustedStaticEval, correctionValue);

    if (eval == VALUE_DRAW)
      eval = value_draw(pos);

    // Can ttValue be used as a better position evaluation?
    if (   ttValue != VALUE_NONE
        && (tte_bound(tte) & (ttValue > eval ? BOUND_LOWER : BOUND_UPPER)))
      eval = ttValue;
  } else {
    unadjustedStaticEval = ss->staticEval = eval = evaluate(pos);

    ss->staticEval = eval = to_corrected_static_eval(unadjustedStaticEval, correctionValue);

    // Static evaluation is saved as it was before adjustment by correction history
    if (!excludedMove)
      tte_save(tte, posKey, VALUE_NONE, ss->ttPv, BOUND_NONE, DEPTH_NONE, 0, unadjustedStaticEval);
  }

  // Use static evaluation difference to improve quiet move ordering (~9 Elo)
  if (   move_is_ok((ss-1)->currentMove)
      && !(ss-1)->checkersBB
      && !captured_piece())
  {
    int bonus = clamp(-16 * ((ss-1)->staticEval + ss->staticEval), -2000, 2000);
    history_update(*pos->mainHistory, !stm(), (ss-1)->currentMove, bonus);
  }

  // Set up the improving flag, which is true if current static evaluation is
  // bigger than the previous static evaluation at our turn (if we were in
  // check at our previous move we go back until we weren't in check) and is
  // false otherwise. The improving flag is used in various pruning heuristics.
  improving = ss->staticEval > (ss - 2)->staticEval;

  opponentWorsening = ss->staticEval + (ss-1)->staticEval > 2;

  // Step 7. Razoring.
  // If eval is really low, check with qsearch if we can exceed alpha. If the
  // search suggests we cannot exceed alpha, return a speculative fail low.
  if (!PvNode && eval < alpha - 469 - 307 * depth * depth)
    return qsearch_NonPV_false(pos, ss, alpha - 1, 0);

  // Step 8. Futility pruning: child node (~50 Elo).
  // The depth condition is important for mate finding.
  if (   !ss->ttPv
      &&  depth < 8
      &&  eval - futility_margin(depth, improving, opponentWorsening) - (ss-1)->statScore / 256 - abs(correctionValue) >= beta
      &&  eval >= beta
      &&  eval < 22266) // 50% larger than VALUE_KNOWN_WIN, but smaller than TB wins.
    return eval; // - futility_margin(depth); (do not do the right thing)

  // Step 9. Null move search with verification search (is omitted in PV nodes)
  if (   !PvNode
      && (ss-1)->currentMove != MOVE_NULL
      && (ss-1)->statScore < 15075
      && eval >= beta
      && eval >= ss->staticEval
      && ss->staticEval >= beta - 23 * depth + 201 - 5 * improving
      && !excludedMove
      && non_pawn_material_c(stm())
      && (ss->ply >= pos->nmpMinPly))
  {
    assert(eval - beta >= 0);

    // Null move dynamic reduction based on depth and value
    Depth R = min((eval - beta) / 189, 6) + depth / 3 + 4;

    ss->currentMove = MOVE_NULL;
    ss->history = &(*pos->counterMoveHistory)[0][0];

    do_null_move(pos);
    ss->endMoves = (ss-1)->endMoves;

    Value nullValue = -search_NonPV(pos, ss+1, -beta, depth-R, !cutNode);
    undo_null_move(pos);

    if (nullValue >= beta) {
      // Do not return unproven mate or TB scores
      nullValue = min(nullValue, VALUE_TB_WIN_IN_MAX_PLY - 1);

      if (pos->nmpMinPly || depth < 14)
        return nullValue;

      assert(!pos->nmpMinPly);

      // Do verification search at high depths with null move pruning
      // disabled until ply exceeds nmpMinPly.
      pos->nmpMinPly = ss->ply + 3 * (depth-R) / 4;

      Value v = search_NonPV(pos, ss, beta-1, depth-R, false);

      pos->nmpMinPly = 0;

      if (v >= beta)
        return nullValue;
    }
  }

  probCutBeta = beta + 170 - 53 * improving - 30 * opponentWorsening;

  // Step 10. ProbCut
  // If we have a good enough capture and a reduced search returns a value
  // much above beta, we can (almost) safely prune the previous move.
  if (   !PvNode
      &&  depth > 3
      &&  abs(beta) < VALUE_TB_WIN_IN_MAX_PLY
      && !(   tte_depth(tte) >= depth - 3
           && ttValue != VALUE_NONE
           && ttValue < probCutBeta))
  {
    mp_init_pc(pos, ttMove, probCutBeta - ss->staticEval);

    while ((move = next_move(pos, 0)))
      if (move != excludedMove && is_legal(pos, move)) {
        assert(is_capture_or_promotion(pos, move));

        captureOrPromotion = true;

        ss->currentMove = move;
        ss->history = &(*pos->counterMoveHistory)[moved_piece(move)][to_sq(move)];
        givesCheck = gives_check(pos, ss, move);
        do_move(pos, move, givesCheck);

        // Perform a preliminary qsearch to verify that the move holds
        value =   givesCheck
               ? -qsearch_NonPV_true(pos, ss+1, -probCutBeta, 0)
               : -qsearch_NonPV_false(pos, ss+1, -probCutBeta, 0);

        // If the qsearch held, perform the regular search
        if (value >= probCutBeta)
          value = -search_NonPV(pos, ss+1, -probCutBeta, depth - 4, !cutNode);

        undo_move(pos, move);
        if (value >= probCutBeta) {
          // Save ProbCut data into transposition table
          tte_save(tte, posKey, value_to_tt(value, ss->ply), ss->ttPv,
              BOUND_LOWER, depth - 3, move, unadjustedStaticEval);
          return value - (probCutBeta - beta);
        }
      }
  }

  // Step 11. Internal iterative reductions (~9 Elo)
  // For PV nodes without a ttMove, we decrease depth.
  if (   PvNode
      && !ttMove)
    depth--;

  // Use qsearch if depth is equal or below zero (~4 Elo)
  if (depth <= 0)
    return qsearch_PV_true(pos, ss, alpha, beta, 0);

  // For cutNodes, if depth is high enough, decrease depth
  if (   cutNode
      && depth >= 8
      && !ttMove)
      depth--;

moves_loop: // When in check search starts from here

  // Step 12. A small Probcut idea, when we are in check
  probCutBeta = beta + 401;
  if (   inCheck
      && !PvNode
      && ttCapture
      && (tte_bound(tte) & BOUND_LOWER)
      && tte_depth(tte) >= depth - 4
      && ttValue >= probCutBeta
      && abs(ttValue) <= VALUE_KNOWN_WIN
      && abs(beta) <= VALUE_KNOWN_WIN
     )
    return probCutBeta;

  PieceToHistory *cmh  = (ss-1)->history;
  PieceToHistory *fmh  = (ss-2)->history;
  PieceToHistory *fmh2 = (ss-4)->history;

  mp_init(pos, ttMove, depth, ss->ply);

  value = bestValue;
  moveCountPruning = false;

  // Indicate PvNodes that will probably fail low if node was searched with
  // non-PV search at depth equal to or greater than current depth and the
  // result of the search was far below alpha
  bool likelyFailLow =   PvNode
                      && ttMove
                      && (tte_bound(tte) & BOUND_UPPER)
                      && tte_depth(tte) >= depth;

  // Step 13. Loop through moves
  // Loop through all pseudo-legal moves until no moves remain or a beta
  // cutoff occurs
  while ((move = next_move(pos, moveCountPruning))) {
    assert(move_is_ok(move));

    if (move == excludedMove)
      continue;

    // At root obey the "searchmoves" option and skip moves not listed
    // inRoot Move List. As a consequence any illegal move is also skipped.
    // In MultiPV mode we also skip PV moves which have been already
    // searched.
    if (rootNode) {
      int idx;
      for (idx = 0; idx < pos->pvLast; idx++)
        if (pos->rootMoves->move[idx].pv[0] == move)
          break;
      if (idx == pos->pvLast)
        continue;
    }

    // Check for legality just before making the move
    if (!rootNode && !is_legal(pos, move))
      continue;

    ss->moveCount = ++moveCount;

    if (rootNode && time_elapsed() > 3000) {
      char buf[16];
      printf("info depth %d currmove %s currmovenumber %d\n",
             depth,
             uci_move(buf, move, is_chess960()),
             moveCount);
      fflush(stdout);
    }

    if (PvNode)
      (ss+1)->pv = NULL;

    extension = 0;
    captureOrPromotion = is_capture_or_promotion(pos, move);
    movedPiece = moved_piece(move);

    givesCheck = gives_check(pos, ss, move);

    // Calculate new depth for this move
    newDepth = depth - 1;

    Value delta = beta - alpha;

    Depth r = reduction(improving, depth, moveCount, delta, pos->rootDelta);

    // Step 14. Pruning at shallow depth
    // Depth conditions are important for mate finding.
    if (  !rootNode
        && non_pawn_material_c(stm())
        && bestValue > VALUE_TB_LOSS_IN_MAX_PLY)
    {
      // Skip quiet moves if movecount exceeds our FutilityMoveCount threshold
      if (!moveCountPruning)
        moveCountPruning = moveCount >= futility_move_count(improving, depth);

      // Reduced depth of the next LMR search
      int lmrDepth = max(newDepth - r / 1024, 0);

      if (   captureOrPromotion
          || givesCheck)
      {
        // Futility pruning for captures (~2 Elo)
        if (   !is_empty(to_sq(move))
            && !givesCheck
            && !PvNode
            && lmrDepth < 7
            && !inCheck
            && ss->staticEval + 424 + 138 * lmrDepth + PieceValue[piece_on(to_sq(move))]
             + (*pos->captureHistory)[movedPiece][to_sq(move)][type_of_p(piece_on(to_sq(move)))] / 7 < alpha)
          continue;

        // SEE based pruning
        if (!see_test(pos, move, -214 * depth))
          continue;

      } else {
        int history =   PIECE_TO_HISTORY_GRAIN * (*cmh)[movedPiece][to_sq(move)]
                      + PIECE_TO_HISTORY_GRAIN * (*fmh)[movedPiece][to_sq(move)]
                      + PIECE_TO_HISTORY_GRAIN * (*fmh2)[movedPiece][to_sq(move)];

        // Continuation history based pruning (~20 Elo)
        if (   lmrDepth < 4
            && history < -3875 * (depth - 1))
          continue;

        history += 2 * 64 * (*pos->mainHistory)[stm()][from_to(move)];

        lmrDepth += history / 16384;

        Value futilityValue =
          ss->staticEval + (bestMove ? 147 : 237) + 125 * lmrDepth + history / 64;

        // Futility pruning: parent node (~13 Elo)
        if (   !inCheck
            && lmrDepth < 11
            && futilityValue <= alpha)
          continue;

        lmrDepth = max(lmrDepth, 0);

        // Prune moves with negative SEE (~20 Elo)
        if (!see_test(pos, move, -25 * lmrDepth * lmrDepth))
          continue;
      }
    }

    // Step 15. Extensions
    // We take care to not overdo to avoid search getting stuck.
    if (ss->ply < pos->rootDepth * 2)
    {
      // Singular extension search. If all moves but one fail low on a search
      // of (alpha-s, beta-s), and just one fails high on (alpha, beta), then
      // that move is singular and should be extended. To verify this we do a
      // reduced search on all the other moves but the ttMove and if the
      // result is lower than ttValue minus a margin, then we extend the ttMove.
      if (    depth >= 6 + 2 * (PvNode && tte_is_pv(tte))
          &&  move == ttMove
          && !rootNode
          && !excludedMove // No recursive singular search
       /* &&  ttValue != VALUE_NONE implicit in the next condition */
          &&  abs(ttValue) < VALUE_KNOWN_WIN
          && (tte_bound(tte) & BOUND_LOWER)
          &&  tte_depth(tte) >= depth - 3)
      {
        Value singularBeta = ttValue - (3 + (ss->ttPv && !PvNode)) * depth;
        Depth singularDepth = (depth - 1) / 2;
        ss->excludedMove = move;
        Move cm = ss->countermove;
        Move k1 = ss->mpKillers[0], k2 = ss->mpKillers[1];
        value = search_NonPV(pos, ss, singularBeta - 1, singularDepth, cutNode);
        ss->excludedMove = 0;

        if (value < singularBeta) {
          extension = 1;

          // Avoid search explosion by limiting the number of double extensions
          if (   !PvNode
              && value < singularBeta - 32
              && ss->doubleExtensions <= 10)
          {
            extension = 2 + (value < singularBeta - 200 && !ttCapture);
            depth += depth < 12;
          }
        }

        // Multi-cut pruning. Our ttMove is assumed to fail high, and now we
        // failed high also on a reduced search without the ttMove. So we
        // assume that this expected cut-node is not singular, i.e. multiple
        // moves fail high. We therefore prune the whole subtree by returning
        // a soft bound.
        else if (value >= beta && value > VALUE_TB_LOSS_IN_MAX_PLY && value < VALUE_TB_WIN_IN_MAX_PLY)
          return value;

        // If the eval of ttMove is greater than beta, we reduce it (negative extension)
        else if (ttValue >= beta) {
          // Fix up our move picker data
          mp_init(pos, ttMove, depth, ss->ply);
          ss->stage++;
          ss->countermove = cm; // pedantic
          ss->mpKillers[0] = k1; ss->mpKillers[1] = k2;

          extension = -2;
        }

        // If the eval of ttMove is less than alpha and value, we reduce it (negative extension)
        else if (ttValue <= alpha) {
          // Fix up our move picker data
          mp_init(pos, ttMove, depth, ss->ply);
          ss->stage++;
          ss->countermove = cm; // pedantic
          ss->mpKillers[0] = k1; ss->mpKillers[1] = k2;

          extension = -1;
        }

        // If the eval of ttMove is less than alpha and value, we reduce it (negative extension)
        else if (cutNode) {
          // Fix up our move picker data
          mp_init(pos, ttMove, depth, ss->ply);
          ss->stage++;
          ss->countermove = cm; // pedantic
          ss->mpKillers[0] = k1; ss->mpKillers[1] = k2;

          extension = -1;
        }

        // The call to search_NonPV with the same value of ss messed up our
        // move picker data. So we fix it.
        mp_init(pos, ttMove, depth, ss->ply);
        ss->stage++;
        ss->countermove = cm; // pedantic
        ss->mpKillers[0] = k1; ss->mpKillers[1] = k2;
      }

      // Check extensions (~1 Elo)
      else if (   givesCheck
               && depth > 8)
        extension = 1;
    }

    // Add extension to new depth
    newDepth += extension;
    ss->doubleExtensions = (ss-1)->doubleExtensions + (extension >= 2);

    // Speculative prefetch as early as possible
    prefetch(tt_first_entry(key_after(pos, move)));

    // Update the current move (this must be done after singular extension
    // search)
    ss->currentMove = move;
    ss->history = &(*pos->counterMoveHistory)[movedPiece][to_sq(move)];

    // Step 16. Make the move.
    do_move(pos, move, givesCheck);

    // HACK: Fix bench after introduction of 2-fold MultiPV bug
    if (rootNode) pos->st[-1].key ^= pos->rootKeyFlip;

    // Decrease reduction if position is or has been on the PV and the node
    // is not likely to fail low
    if (ss->ttPv && !likelyFailLow)
      r -= 2048;

    // Decrease reduction if opponent's move count is high
    if ((ss-1)->moveCount > 7)
      r -= 1024;

    // printf("correctionValue: %d\n", abs(correctionValue / 32));
    r -= abs(correctionValue) * 4;

    // Increase reduction for cut nodes
    if (cutNode)
      r += 2048;

    // Increase reduction if ttMove is a capture (~3 Elo)
    if (ttCapture)
      r += 1024;

    // Decrease reduction at PvNodes if bestvalue
    // is vastly different from static evaluation
    // if (PvNode && !inCheck && abs(ss->staticEval - bestValue) > 250)
    //   r--;

    // Decrease reduction for PvNodes (~2 Elo)
    if (PvNode)
      r -= 1024;

    // Increase reduction if next ply has a lot of fail high else reset count to 0
    if ((ss+1)->cutoffCnt > 3)
      r += 1024;

    ss->statScore =  2 * PIECE_TO_HISTORY_GRAIN * (*cmh )[movedPiece][to_sq(move)]
                   + PIECE_TO_HISTORY_GRAIN * (*fmh )[movedPiece][to_sq(move)]
                   + PIECE_TO_HISTORY_GRAIN * (*fmh2)[movedPiece][to_sq(move)]
                   + 64 * (*pos->mainHistory)[!stm()][from_to(move)]
                   - 4123;

    // Decrease/increase reduction for moves with a good/bad history (~30 Elo)
    r -= (ss->statScore / (10500 + 4500 * (depth > 7 && depth < 19))) * 1024;

    // Step 17. Late moves reduction / extension (LMR)
    // We use various heuristics for the children of a node after the first
    // child has been searched. In general we would like to reduce them, but
    // there are many cases where we extend a child if it has good chances
    // to be "interesting".
    if (    depth >= 2
        &&  moveCount > 1 + (PvNode && ss->ply <= 1)
        && (   !ss->ttPv
            || !captureOrPromotion
            || (cutNode && (ss-1)->moveCount > 1)))
    {
      // In general we want to cap the LMR depth search at newDepth, but when
      // reduction is negative, we allow this move a limited search extension
      // beyond the first move depth. This may lead to hidden double extensions.
      Depth d = clamp(newDepth - r / 1024, 1, newDepth + 1);

      value = -search_NonPV(pos, ss+1, -(alpha+1), d, 1);

      // Do full depth search when reduced LMR search fails high
      if (value > alpha && d < newDepth)
      {
        // Adjust full depth search based on LMR results - if result
        // was good enough search deeper, if it was bad enough search shallower
        const bool doDeeperSearch = value > (bestValue + 42 + 2 * newDepth);
        const bool doShallowerSearch = value < bestValue + newDepth;

        newDepth += doDeeperSearch - doShallowerSearch;
        if (newDepth > d)
          value = -search_NonPV(pos, ss+1, -(alpha+1), newDepth, !cutNode);

        int bonus = value <= alpha ? -stat_malus(newDepth)
                  : value >= beta  ?  stat_bonus(newDepth)
                                   :  0;

        update_continuation_histories(ss, movedPiece, to_sq(move), bonus);
      }
    }

    // Step 18. Full depth search when LMR is skipped. If expected reduction is high, reduce its depth by 1.
    else if  (!PvNode || moveCount > 1)
    {
      // value = -search_NonPV(pos, ss+1, -(alpha+1), newDepth - (r > 4), !cutNode);
      value = -search_NonPV(pos, ss+1, -(alpha+1), newDepth, !cutNode);
    }

    // For PV nodes only, do a full PV search on the first move or after a fail
    // high (in the latter case search only if value < beta), otherwise let the
    // parent node fail low with value <= alpha and try another move.
    if (   PvNode
        && (moveCount == 1 || (value > alpha && (rootNode || value < beta))))
    {
      (ss+1)->pv = pv;
      (ss+1)->pv[0] = 0;

      value = -search_PV(pos, ss+1, -beta, -alpha, newDepth);
    }

    // Step 19. Undo move
    // HACK: Fix bench after introduction of 2-fold MultiPV bug
    if (rootNode) pos->st[-1].key ^= pos->rootKeyFlip;
    undo_move(pos, move);

    assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

    // Step 20. Check for a new best move
    // Finished searching the move. If a stop occurred, the return value of
    // the search cannot be trusted, and we return immediately without
    // updating best move, PV and TT.
    if (load_rlx(Threads.stop))
      return 0;

    if (rootNode) {
      RootMove *rm = NULL;
      for (int idx = 0; idx < pos->rootMoves->size; idx++)
        if (pos->rootMoves->move[idx].pv[0] == move) {
          rm = &pos->rootMoves->move[idx];
          break;
        }

      rm->averageScore = rm->averageScore != -VALUE_INFINITE ? (2 * value + rm->averageScore) / 3 : value;
      // PV move or new best move ?
      if (moveCount == 1 || value > alpha) {
        rm->score = value;
        rm->selDepth = pos->selDepth;
        rm->pvSize = 1;

        assert((ss+1)->pv);

        for (Move *m = (ss+1)->pv; *m; ++m)
          rm->pv[rm->pvSize++] = *m;

        // We record how often the best move has been changed in each
        // iteration. This information is used for time management: When
        // the best move changes frequently, we allocate some more time.
        if (moveCount > 1)
          pos->bestMoveChanges++;
      } else
        // All other moves but the PV are set to the lowest value: this is
        // not a problem when sorting because the sort is stable and the
        // move position in the list is preserved - just the PV is pushed up.
        rm->score = -VALUE_INFINITE;
    }

    if (value > bestValue) {
      bestValue = value;

      if (value > alpha) {
        bestMove = move;

        if (PvNode && !rootNode) // Update pv even in fail-high case
          update_pv(ss->pv, move, (ss+1)->pv);

        if (value >= beta)
        {
          if (ss->cutoffCnt < 254)
            ss->cutoffCnt += 1 + !ttMove;
          assert(value >= beta); // Fail high
          break;
        }
        else
        {
          // Reduce other moves if we have found at least one score improvement (~1 Elo)
          if (   depth > 1
              && beta  <  12535
              && value > -12535)
            depth -= 1;

          assert(depth > 0);
          alpha = value;
        }
      }
    }

    // If the move is worse than some previously searched move, remember it, to update its stats later
    if (move != bestMove && moveCount <= 32) {
      if (captureOrPromotion)
        capturesSearched[captureCount++] = move;

      else
        quietsSearched[quietCount++] = move;
    }
  }

  // The following condition would detect a stop only after move loop has
  // been completed. But in this case bestValue is valid because we have
  // fully searched our subtree, and we can anyhow save the result in TT.
  /*
  if (Threads.stop)
    return VALUE_DRAW;
  */

  // Step 21. Check for mate and stalemate
  // All legal moves have been searched and if there are no legal moves,
  // it must be a mate or a stalemate. If we are in a singular extension
  // search then return a fail low score.
  if (!moveCount)
    bestValue = excludedMove ? alpha
               :     inCheck ? mated_in(ss->ply) : VALUE_DRAW;

  // If there is a move that produces search value greater than alpha,
  // we update the stats of searched moves.
  else if (bestMove) {
    // update_all_stats

    if (!is_capture_or_promotion(pos, bestMove)) {
      int bonus =  bestValue > beta + 140
                 ? stat_bonus(depth + 1)
                 : stat_bonus(depth);
      int malus =  bestValue > beta + 140
                 ? stat_malus(depth + 1)
                 : stat_malus(depth);
      update_quiet_histories(pos, ss, bestMove, bonus * 873 / 1024);

      // Decrease stats for all non-best quiet moves
      for (int i = 0; i < quietCount; i++) {
        history_update(*pos->mainHistory, stm(), quietsSearched[i], -malus * 1185 / 1024);
        update_continuation_histories(ss, moved_piece(quietsSearched[i]),
            to_sq(quietsSearched[i]), -malus * 1189 / 1024);
      }
    }

    // update capture stats
    Piece moved_piece = moved_piece(bestMove);
    PieceType captured = type_of_p(piece_on(to_sq(bestMove)));

    int bonus = stat_bonus(depth);
    int malus = stat_malus(depth);

    if (is_capture_or_promotion(pos, bestMove))
    {
      // Increase stats for the best move in case it was a capture move
      cpth_update(*pos->captureHistory, moved_piece, to_sq(bestMove), captured, bonus * 839 / 1024);
    }

    // Extra penalty for a quiet early move that was not a TT move or main
    // killer move in previous ply when it gets refuted
    if (  ((ss-1)->moveCount == 1 + (ss-1)->ttHit || (ss-1)->currentMove == (ss-1)->killers[0])
        && !captured_piece())
      update_continuation_histories(ss-1, piece_on(prevSq), prevSq, -malus * 993 / 1024);

    // Decrease stats for all non-best capture moves
    for (int i = 0; i < captureCount; i++) {
      moved_piece = moved_piece(capturesSearched[i]);
      captured = type_of_p(piece_on(to_sq(capturesSearched[i])));
      cpth_update(*pos->captureHistory, moved_piece, to_sq(capturesSearched[i]), captured, -malus * 1040 / 1024);
    }
  }

  // Bonus for prior countermove that caused the fail low
  else if (!captured_piece() && prevSq != SQ_NONE)
  {
    int bonus = (117 * (depth > 5) + 39 * (PvNode || cutNode) + 168 * ((ss - 1)->moveCount > 8)
                 + 115 * (!ss->checkersBB && bestValue <= ss->staticEval - 108)
                 + 119 * (!(ss - 1)->checkersBB && bestValue <= -(ss - 1)->staticEval - 83));

    // Proportional to "how much damage we have to undo"
    bonus += min(-(ss - 1)->statScore / 113, 300);
    bonus = max(bonus, 0);

    update_continuation_histories(ss-1, piece_on(prevSq), prevSq, stat_bonus(depth) * bonus / 160);
    history_update(*pos->mainHistory, !stm(), from_to((ss - 1)->currentMove), stat_bonus(depth) * bonus / 307);
  }

  if (PvNode)
    bestValue = min(bestValue, maxValue);

  // If no good move is found and the previous position was ttPv, then the
  // previous opponent move is probably good and the new position is added
  // to the search tree
  if (bestValue <= alpha)
    ss->ttPv = ss->ttPv || (ss-1)->ttPv;

  // Otherwise, a countermove has been found and if the position is in the
  // last leaf in the search tree, remove the position from the search tree.
  else if (depth > 3)
    ss->ttPv = ss->ttPv && (ss+1)->ttPv;

  // Write gathered information in transposition table. Note that the
  // static evaluation is saved as it was before correction history.
  if (!excludedMove)
    tte_save(tte, posKey, value_to_tt(bestValue, ss->ply), ss->ttPv,
        bestValue >= beta ? BOUND_LOWER :
        PvNode && bestMove ? BOUND_EXACT : BOUND_UPPER,
        depth, bestMove, unadjustedStaticEval);

  // Adjust correction history
  if (   !inCheck
      && (!bestMove || !is_capture_or_promotion(pos, bestMove))
      && !(bestValue >= beta && bestValue <= ss->staticEval)
      && !(!bestMove && bestValue >= ss->staticEval))
  {
    int bonus = clamp((bestValue - ss->staticEval) * depth / 8, -256, 256);
    correction_history_update(*pos->correctionHistory, stm(), pos, bonus);
    non_pawn_correction_history_update(*pos->nonPawnCorrectionHistory, WHITE, stm(), pos, bonus);
    non_pawn_correction_history_update(*pos->nonPawnCorrectionHistory, BLACK, stm(), pos, bonus);
  }

  assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);

  return bestValue;
}

// search_PV() is the main search function for PV nodes
static NOINLINE Value search_PV(Position *pos, Stack *ss, Value alpha,
    Value beta, Depth depth)
{
  return search_node(pos, ss, alpha, beta, depth, 0, PV);
}

// search_NonPV is the main search function for non-PV nodes
static NOINLINE Value search_NonPV(Position *pos, Stack *ss, Value alpha,
    Depth depth, bool cutNode)
{
  return search_node(pos, ss, alpha, alpha+1, depth, cutNode, NonPV);
}

// qsearch_node() is the quiescence search function template, which is
// called by the main search function with zero depth, or recursively with
// further decreasing depth per call.
INLINE Value qsearch_node(Position *pos, Stack *ss, Value alpha, Value beta,
    Depth depth, const int NT, const bool InCheck)
{
  const bool PvNode = NT == PV;

  assert(InCheck == (bool)checkers());
  assert(alpha >= -VALUE_INFINITE && alpha < beta && beta <= VALUE_INFINITE);
  assert(PvNode || (alpha == beta - 1));
  assert(depth <= 0);

  // Check if we have an upcoming move that draws by repetition, or
  // if the opponent had an alternative move earlier to this position.
  if (   depth < 0
      && rule50_count() >= 3
      && alpha < VALUE_DRAW
      && has_game_cycle(pos, ss->ply))
  {
    alpha = value_draw(pos);
    if (alpha >= beta)
      return alpha;
  }

  Move pv[MAX_PLY+1];
  TTEntry *tte;
  Key posKey;
  Move ttMove, move, bestMove;
  Value bestValue, value, ttValue, futilityValue, futilityBase;
  bool pvHit, givesCheck;
  Depth ttDepth;
  int moveCount;

  if (PvNode) {
    (ss+1)->pv = pv;
    ss->pv[0] = 0;
  }

  bestMove = 0;
  moveCount = 0;

  // Check for an instant draw or if the maximum ply has been reached
  if (is_draw(pos) || ss->ply >= MAX_PLY)
    return ss->ply >= MAX_PLY && !InCheck ? evaluate(pos) : VALUE_DRAW;

  assert(0 <= ss->ply && ss->ply < MAX_PLY);

  // Decide whether or not to include checks: this fixes also the type of
  // TT entry depth that we are going to use. Note that in qsearch we use
  // only two types of depth in TT: DEPTH_QS_CHECKS or DEPTH_QS_NO_CHECKS.
  ttDepth = InCheck || depth >= DEPTH_QS_CHECKS ? DEPTH_QS_CHECKS
                                                : DEPTH_QS_NO_CHECKS;

  // Transposition table lookup
  posKey = key();
  tte = tt_probe(posKey, &ss->ttHit);
  ttValue = ss->ttHit ? value_from_tt(tte_value(tte), ss->ply, rule50_count()) : VALUE_NONE;
  ttMove = ss->ttHit ? tte_move(tte) : 0;
  pvHit = ss->ttHit && tte_is_pv(tte);

  if (  !PvNode
      && ss->ttHit
      && tte_depth(tte) >= ttDepth
      && ttValue != VALUE_NONE // Only in case of TT access race
      && (ttValue >= beta ? (tte_bound(tte) &  BOUND_LOWER)
                          : (tte_bound(tte) &  BOUND_UPPER)))
    return ttValue;

  Value unadjustedStaticEval = VALUE_NONE;

  // Evaluate the position statically
  int correctionValue = correction_value(pos);
  if (InCheck) {
    bestValue = futilityBase = -VALUE_INFINITE;
  } else {
    if (ss->ttHit) {
      // Never assume anything about values stored in TT
      if ((unadjustedStaticEval = ss->staticEval = bestValue = tte_eval(tte)) == VALUE_NONE)
         unadjustedStaticEval = ss->staticEval = bestValue = evaluate(pos);

      ss->staticEval = bestValue = to_corrected_static_eval(unadjustedStaticEval, correctionValue);

      // Can ttValue be used as a better position evaluation?
      if (    ttValue != VALUE_NONE
          && (tte_bound(tte) & (ttValue > bestValue ? BOUND_LOWER : BOUND_UPPER)))
        bestValue = ttValue;
    } else
    {
      unadjustedStaticEval = ss->staticEval = bestValue =
        (ss-1)->currentMove != MOVE_NULL ? evaluate(pos)
                                         : -(ss-1)->staticEval;

      ss->staticEval = bestValue = to_corrected_static_eval(unadjustedStaticEval, correctionValue);
    }

    // Stand pat. Return immediately if static value is at least beta
    if (bestValue >= beta) {
      if (!ss->ttHit)
        tte_save(tte, posKey, value_to_tt(bestValue, ss->ply), false,
            BOUND_LOWER, DEPTH_NONE, 0, unadjustedStaticEval);

      return bestValue;
    }

    if (bestValue > alpha)
      alpha = bestValue;

    futilityBase = ss->staticEval + 139;
  }

  ss->history = &(*pos->counterMoveHistory)[0][0];

  // Initialize move picker data for the current position, and prepare
  // to search the moves. Because the depth is <= 0 here, only captures,
  // queen promotions and checks (only if depth >= DEPTH_QS_CHECKS) will
  // be generated.
  Square prevSq = move_is_ok((ss-1)->currentMove) ? to_sq((ss-1)->currentMove) : SQ_NONE;
  mp_init_q(pos, ttMove, depth, prevSq);

  // Loop through the moves until no moves remain or a beta cutoff occurs
  while ((move = next_move(pos, 0))) {
    assert(move_is_ok(move));

    // Check for legality
    if (!is_legal(pos, move)) {
      continue;
    }

    givesCheck = gives_check(pos, ss, move);

    moveCount++;

    // Futility pruning and moveCount pruning (~5 Elo)
    if (    bestValue > VALUE_TB_LOSS_IN_MAX_PLY
        && !givesCheck
        &&  to_sq(move) != prevSq
        &&  futilityBase > -VALUE_KNOWN_WIN
        &&  type_of_m(move) != PROMOTION)
    {
      if (moveCount > 2)
        continue;

      futilityValue = futilityBase + PieceValue[piece_on(to_sq(move))];

      if (futilityValue <= alpha) {
        bestValue = max(bestValue, futilityValue);
        continue;
      }

      if (futilityBase <= alpha && !see_test(pos, move, 1)) {
        bestValue = max(bestValue, futilityBase);
        continue;
      }
    }

    // Continuation history based pruning (~3 Elo) - why does adding this work?
    if (   !is_capture_or_promotion(pos, move)
        && PIECE_TO_HISTORY_GRAIN * (*(ss-1)->history)[moved_piece(move)][to_sq(move)] < CounterMovePruneThreshold
        && PIECE_TO_HISTORY_GRAIN * (*(ss-2)->history)[moved_piece(move)][to_sq(move)] < CounterMovePruneThreshold)
      continue;

    // Do not search moves with bad enough SEE values (~5 Elo)
    if (!see_test(pos, move, -83))
      continue;

    // Speculative prefetch as early as possible
    prefetch(tt_first_entry(key_after(pos, move)));

    ss->currentMove = move;
    ss->history = &(*pos->counterMoveHistory)[moved_piece(move)]
                                             [to_sq(move)];

    // Make and search the move
    do_move(pos, move, givesCheck);
    value = PvNode ? givesCheck
                     ? -qsearch_PV_true(pos, ss+1, -beta, -alpha, depth - 1)
                     : -qsearch_PV_false(pos, ss+1, -beta, -alpha, depth - 1)
                   : givesCheck
                     ? -qsearch_NonPV_true(pos, ss+1, -beta, depth - 1)
                     : -qsearch_NonPV_false(pos, ss+1, -beta, depth - 1);
    undo_move(pos, move);

    assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

    // Check for a new best move
    if (value > bestValue) {
      bestValue = value;

      if (value > alpha) {
        bestMove = move;

        if (PvNode) // Update pv even in fail-high case
          update_pv(ss->pv, move, (ss+1)->pv);

        if (value < beta) // Update alpha here!
          alpha = value;
        else
          break; // Fail high
      }
    }
  }

  // All legal moves have been searched. A special case: If we're in check
  // and no legal moves were found, it is checkmate.
  if (InCheck && bestValue == -VALUE_INFINITE)
    return mated_in(ss->ply); // Plies to mate from the root

  tte_save(tte, posKey, value_to_tt(bestValue, ss->ply), pvHit,
      bestValue >= beta ? BOUND_LOWER : BOUND_UPPER,
      ttDepth, bestMove, unadjustedStaticEval);

  assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);

  return bestValue;
}

static NOINLINE Value qsearch_PV_true(Position *pos, Stack *ss, Value alpha,
    Value beta, Depth depth)
{
  return qsearch_node(pos, ss, alpha, beta, depth, PV, true);
}

static NOINLINE Value qsearch_PV_false(Position *pos, Stack *ss, Value alpha,
    Value beta, Depth depth)
{
  return qsearch_node(pos, ss, alpha, beta, depth, PV, false);
}

static NOINLINE Value qsearch_NonPV_true(Position *pos, Stack *ss, Value alpha,
    Depth depth)
{
  return qsearch_node(pos, ss, alpha, alpha+1, depth, NonPV, true);
}

static NOINLINE Value qsearch_NonPV_false(Position *pos, Stack *ss, Value alpha,
    Depth depth)
{
  return qsearch_node(pos, ss, alpha, alpha+1, depth, NonPV, false);
}

#define rm_lt(m1,m2) ((m1).score != (m2).score ? (m1).score < (m2).score : (m1).previousScore < (m2).previousScore)

// stable_sort() sorts RootMoves from highest-scoring move to lowest-scoring
// move while preserving order of equal elements.
static void stable_sort(RootMove *rm, int num)
{
  int i, j;

  for (i = 1; i < num; i++)
    if (rm_lt(rm[i - 1], rm[i])) {
      RootMove tmp = rm[i];
      rm[i] = rm[i - 1];
      for (j = i - 1; j > 0 && rm_lt(rm[j - 1], tmp); j--)
        rm[j] = rm[j - 1];
      rm[j] = tmp;
    }
}

// value_to_tt() adjusts a mate score from "plies to mate from the root" to
// "plies to mate from the current position". Non-mate scores are unchanged.
// The function is called before storing a value in the transposition table.

static Value value_to_tt(Value v, int ply)
{
  assert(v != VALUE_NONE);

  return  v >= VALUE_TB_WIN_IN_MAX_PLY  ? v + ply
        : v <= VALUE_TB_LOSS_IN_MAX_PLY ? v - ply : v;
}


// value_from_tt() is the inverse of value_to_tt(): It adjusts a mate score
// from the transposition table (which refers to the plies to mate/be mated
// from current position) to "plies to mate/be mated from the root".

static Value value_from_tt(Value v, int ply, int r50c)
{
  if (v == VALUE_NONE)
    return VALUE_NONE;

  if (v >= VALUE_TB_WIN_IN_MAX_PLY) {
    if (v >= VALUE_MATE_IN_MAX_PLY && VALUE_MATE - v > 99 - r50c)
      return VALUE_MATE_IN_MAX_PLY - 1;
    return v - ply;
  }

  if (v <= VALUE_TB_LOSS_IN_MAX_PLY) {
    if (v <= VALUE_MATED_IN_MAX_PLY && VALUE_MATE + v > 99 - r50c)
      return VALUE_MATED_IN_MAX_PLY + 1;
    return v + ply;
  }

  return v;
}


// update_pv() adds current move and appends child pv[]

static void update_pv(Move *pv, Move move, Move *childPv)
{
  for (*pv++ = move; childPv && *childPv; )
    *pv++ = *childPv++;
  *pv = 0;
}


// update_continuation_histories() updates countermove and follow-up move history.

static void update_continuation_histories(Stack *ss, Piece pc, Square s, int bonus)
{
  if (move_is_ok((ss-1)->currentMove))
    cms_update(*(ss-1)->history, pc, s, bonus * 949 / 1024);

  if (move_is_ok((ss-2)->currentMove))
    cms_update(*(ss-2)->history, pc, s, bonus * 1003 / 1024);

  if (ss->checkersBB)
    return;

  if (move_is_ok((ss-3)->currentMove))
    cms_update(*(ss-3)->history, pc, s, bonus * 149 / 256);

  if (move_is_ok((ss-4)->currentMove))
    cms_update(*(ss-4)->history, pc, s, bonus * 1001 / 1024);

  if (move_is_ok((ss-5)->currentMove))
    cms_update(*(ss-5)->history, pc, s, bonus * 122 / 1024);

  if (move_is_ok((ss-6)->currentMove))
    cms_update(*(ss-6)->history, pc, s, bonus * 977 / 1024);
}

// update_quiet_histories() updates killers, history, countermove and countermove
// plus follow-up move history when a new quiet best move is found.

static void update_quiet_histories(const Position *pos, Stack *ss, Move move, int bonus)
{
  if (ss->killers[0] != move) {
    ss->killers[1] = ss->killers[0];
    ss->killers[0] = move;
  }

  Color c = stm();
  history_update(*pos->mainHistory, c, move, bonus);
  update_continuation_histories(ss, moved_piece(move), to_sq(move), bonus);

  if (move_is_ok((ss-1)->currentMove)) {
    Square prevSq = to_sq((ss-1)->currentMove);
    (*pos->counterMoves)[piece_on(prevSq)][prevSq] = move;
  }
}


// check_time() is used to print debug info and, more importantly, to detect
// when we are out of available time and thus stop the search.

static void check_time(void)
{
  TimePoint elapsed = time_elapsed();

  // An engine may not stop pondering until told so by the GUI
  if (Threads.ponder)
    return;

  if (   (use_time_management() && elapsed > time_maximum() - 10)
      || (Limits.movetime && elapsed >= Limits.movetime))
        Threads.stop = 1;
}

// uci_print_pv() prints PV information according to the UCI protocol.
// UCI requires that all (if any) unsearched PV lines are sent with a
// previous search score.

static void uci_print_pv(Position *pos, Depth depth, Value alpha, Value beta)
{
  TimePoint elapsed = time_elapsed() + 1;
  RootMoves *rm = pos->rootMoves;
  uint64_t nodes_searched = threads_nodes_searched();
  char buf[16];

  flockfile(stdout);
  const int i = 0;

  bool updated = rm->move[i].score != -VALUE_INFINITE;

  Depth d = updated ? depth : max(1, depth - 1);
  Value v = updated ? rm->move[i].score : rm->move[i].previousScore;

  if (v == -VALUE_INFINITE)
    v = VALUE_ZERO;

  printf("info depth %d seldepth %d score %s",
         d, rm->move[i].selDepth + 1,
         uci_value(buf, v));

  printf("%s", v >= beta ? " lowerbound" : v <= alpha ? " upperbound" : "");

  printf(" nodes %"PRIu64" nps %"PRIu64, nodes_searched,
                            nodes_searched * 1000 / elapsed);

#ifndef MINIMAL
  if (elapsed > 1000)
    printf(" hashfull %d", tt_hashfull());
#endif

  printf(" time %"PRIi64" pv", elapsed);

  for (int idx = 0; idx < rm->move[i].pvSize; idx++)
    printf(" %s", uci_move(buf, rm->move[i].pv[idx], is_chess960()));
  printf("\n");

  fflush(stdout);
  funlockfile(stdout);
}


// extract_ponder_from_tt() is called in case we have no ponder move
// before exiting the search, for instance, in case we stop the search
// during a fail high at root. We try hard to have a ponder move to
// return to the GUI, otherwise in case of 'ponder on' we have nothing
// to think on.

static int extract_ponder_from_tt(RootMove *rm, Position *pos)
{
  bool ttHit;

  assert(rm->pvSize == 1);

  if (!rm->pv[0])
    return 0;

  do_move(pos, rm->pv[0], gives_check(pos, pos->st, rm->pv[0]));
  TTEntry *tte = tt_probe(key(), &ttHit);

  if (ttHit) {
    Move m = tte_move(tte); // Local copy to be SMP safe
    ExtMove list[MAX_MOVES];
    ExtMove *last = generate_legal(pos, list);
    for (ExtMove *p = list; p < last; p++)
      if (p->move == m) {
        rm->pv[rm->pvSize++] = m;
        break;
      }
  }

  undo_move(pos, rm->pv[0]);
  return rm->pvSize > 1;
}


// start_thinking() wakes up the main thread to start a new search,
// then returns immediately.

void start_thinking(Position *root, bool ponderMode)
{
  if (Threads.searching)
    thread_wait_until_sleeping(threads_main());

  Threads.stopOnPonderhit = false;
  Threads.stop = false;
  Threads.increaseDepth = true;
  Threads.ponder = ponderMode;

  // Generate all legal moves.
  ExtMove list[MAX_MOVES];
  ExtMove *end = generate_legal(root, list);

  RootMoves *moves = Threads.pos->rootMoves;
  moves->size = end - list;
  for (int i = 0; i < moves->size; i++)
    moves->move[i].pv[0] = list[i].move;

  Position *pos = Threads.pos;
  pos->selDepth = 0;
  pos->nmpMinPly = 0;
  pos->rootDepth = 0;
  pos->nodes = 0;

  RootMoves *rm = pos->rootMoves;
  rm->size = end - list;
  for (int i = 0; i < rm->size; i++) {
    rm->move[i].pvSize = 1;
    rm->move[i].pv[0] = moves->move[i].pv[0];
    rm->move[i].score = -VALUE_INFINITE;
    rm->move[i].previousScore = -VALUE_INFINITE;
    rm->move[i].selDepth = 0;
  }
  memcpy(pos, root, offsetof(Position, moveList));
  // Copy enough of the root State buffer.
  int n = max(7, root->st->pliesFromNull);
  for (int i = 0; i <= n; i++)
    memcpy(&pos->stack[i], &root->st[i - n], StateSize);
  pos->st = pos->stack + n;
  (pos->st-1)->endMoves = pos->moveList;
  pos_set_check_info(pos);

  // full refresh
  nnue_accumulator_refresh(&pos->st->accumulator, pos, WHITE);
  nnue_accumulator_refresh(&pos->st->accumulator, pos, BLACK);
  // accumulator->computed[WHITE] = true;
  // accumulator->computed[BLACK] = true;

  Threads.searching = true;
  thread_wake_up(threads_main(), THREAD_SEARCH);
}
