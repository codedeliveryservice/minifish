/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2017 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad

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

#include "movegen.h"
#include "movepick.h"
#include "search.h"
#include "settings.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"

static void thread_idle_loop(Position *pos);

#define THREAD_FUNC void *

// Global objects
ThreadPool Threads;
MainThread mainThread;

// thread_init() is where a search thread starts and initialises itself.

static THREAD_FUNC thread_init()
{
  Position *pos;

  pos = calloc(sizeof(Position), 1);
  pos->counterMoves = calloc(sizeof(CounterMoveStat), 1);
  pos->mainHistory = calloc(sizeof(ButterflyHistory), 1);
  pos->captureHistory = calloc(sizeof(CapturePieceToHistory), 1);
  pos->correctionHistory = calloc(sizeof(CorrectionHistory), 1);
  pos->nonPawnCorrectionHistory = calloc(sizeof(NonPawnCorrectionHistory), 1);

  pos->rootMoves = calloc(sizeof(RootMoves), 1);
  pos->stackAllocation = calloc(63 + (MAX_PLY + 50) * sizeof(Stack), 1);
  pos->moveList = calloc(10000 * sizeof(ExtMove), 1);

  pos->stack = (Stack *)(((uintptr_t)pos->stackAllocation + 0x3f) & ~0x3f);
  pos->counterMoveHistory = calloc(sizeof(CounterMoveHistoryStat), 1);
  for (int j = 0; j < 16; j++)
    for (int k = 0; k < 64; k++)
      (*pos->counterMoveHistory)[0][0][j][k] = CounterMovePruneThreshold - 1;

  atomic_store(&pos->resetCalls, false);
  pos->selDepth = pos->callsCnt = 0;

  pthread_mutex_init(&pos->mutex, NULL);
  pthread_cond_init(&pos->sleepCondition, NULL);

  Threads.pos = pos;

  pthread_mutex_lock(&Threads.mutex);
  Threads.initializing = false;
  pthread_cond_signal(&Threads.sleepCondition);
  pthread_mutex_unlock(&Threads.mutex);

  thread_idle_loop(pos);

  return 0;
}

// thread_create() launches a new thread.

static void thread_create()
{
  pthread_t thread;

  Threads.initializing = true;
  pthread_mutex_lock(&Threads.mutex);
  pthread_create(&thread, NULL, thread_init, NULL);
  while (Threads.initializing)
    pthread_cond_wait(&Threads.sleepCondition, &Threads.mutex);
  pthread_mutex_unlock(&Threads.mutex);

  Threads.pos->nativeThread = thread;
}


// thread_destroy() waits for thread termination before returning.

static void thread_destroy(Position *pos)
{
  pthread_mutex_lock(&pos->mutex);
  pos->action = THREAD_EXIT;
  pthread_cond_signal(&pos->sleepCondition);
  pthread_mutex_unlock(&pos->mutex);
  pthread_join(pos->nativeThread, NULL);
  pthread_cond_destroy(&pos->sleepCondition);
  pthread_mutex_destroy(&pos->mutex);

  free(pos->counterMoves);
  free(pos->mainHistory);
  free(pos->captureHistory);
  free(pos->counterMoveHistory);

  free(pos->rootMoves);
  free(pos->stackAllocation);
  free(pos->moveList);
  free(pos);
}


// thread_wait_for_search_finished() waits on sleep condition until
// not searching.

void thread_wait_until_sleeping(Position *pos)
{
  pthread_mutex_lock(&pos->mutex);

  while (pos->action != THREAD_SLEEP)
    pthread_cond_wait(&pos->sleepCondition, &pos->mutex);

  pthread_mutex_unlock(&pos->mutex);

  Threads.searching = false;
}


// thread_wait() waits on sleep condition until condition is true.

void thread_wait(Position *pos, atomic_bool *condition)
{
  pthread_mutex_lock(&pos->mutex);

  while (!atomic_load(condition))
    pthread_cond_wait(&pos->sleepCondition, &pos->mutex);

  pthread_mutex_unlock(&pos->mutex);
}


void thread_wake_up(Position *pos, int action)
{
  pthread_mutex_lock(&pos->mutex);

  if (action != THREAD_RESUME)
    pos->action = action;

  pthread_cond_signal(&pos->sleepCondition);
  pthread_mutex_unlock(&pos->mutex);
}


// thread_idle_loop() is where the thread is parked when it has no work to do.

static void thread_idle_loop(Position *pos)
{
  while (true) {
    pthread_mutex_lock(&pos->mutex);

    while (pos->action == THREAD_SLEEP) {
      pthread_cond_signal(&pos->sleepCondition); // Wake up any waiting thread
      pthread_cond_wait(&pos->sleepCondition, &pos->mutex);
    }

    pthread_mutex_unlock(&pos->mutex);

    if (pos->action == THREAD_EXIT) {
      break;

    } else if (pos->action == THREAD_TT_CLEAR) {
      tt_clear_worker(0);

    } else {
      mainthread_search();
    }

    pos->action = THREAD_SLEEP;
  }
}


// threads_init() creates and launches requested threads that will go
// immediately to sleep. We cannot use a constructor because Threads is a
// static object and we need a fully initialized engine at this point due to
// allocation of Endgames in the Thread constructor.

void threads_init(void)
{
  pthread_mutex_init(&Threads.mutex, NULL);
  pthread_cond_init(&Threads.sleepCondition, NULL);

  Threads.numThreads = 1;
  thread_create();
}


// threads_exit() terminates threads before the program exits. Cannot be
// done in destructor because threads must be terminated before deleting
// any static objects while still in main().

void threads_exit(void)
{
  threads_set_number(0);
  pthread_cond_destroy(&Threads.sleepCondition);
  pthread_mutex_destroy(&Threads.mutex);
}


// threads_set_number() creates/destroys threads to match the requested
// number.

void threads_set_number(int num)
{
  if (num == 0)
    thread_destroy(Threads.pos);

  search_init();

  if (num == 0)
    Threads.searching = false;
}


// threads_nodes_searched() returns the number of nodes searched.

uint64_t threads_nodes_searched(void)
{
  return Threads.pos->nodes;
}
