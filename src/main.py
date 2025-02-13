from subprocess import run, Popen, PIPE
from time import time

d = "/kaggle_simulations/agent/"
b = f"{d}e"
run(["7z","x",f"{b}.7z",f"-o{d}","-y"])
e = Popen([b],stdin=PIPE,stdout=PIPE,text=True)
m = []
f = p = None

def k(s):
  return s*1000

def w(t):
  print(f"send: {t.strip()}")
  e.stdin.write(t+"\n")
  e.stdin.flush()

def l():
  for r in e.stdout:
    print(r.strip())
    if r.startswith("bestmove"):
      t = r.strip().split()
      return (t[1], t[3] if len(t) == 4 else None)

def a(o):
  global f, p
  if not f:
    f = o.board
  elif " 0 " in o.board:
    f = o.board
    m.clear()
  elif o.lastMove:
    m.append(o.lastMove)
  t = [k(o.remainingOverageTime), k(o.opponentRemainingOverageTime)]
  if o.mark == "white":
    wt, bt = t
  else:
    bt, wt = t
  t0 = time()
  if p == o.lastMove:
    w("ponderhit")
    b, p = l()
  else:
    if p:
      w("stop")
      l()
    w(f"position fen {f} {'moves '+' '.join(m) if m else ''}")
    w(f"go wtime {wt} winc 0 btime {bt} binc 0")
    b, p = l()
  m.append(b)
  t = k(time()-t0)
  if o.mark == "white":
    wt -= t
  else:
    bt -= t
  if p:
    w(f"position fen {f} moves {' '.join(m)} {p}")
    w(f"go ponder wtime {wt} winc 0 btime {bt} binc 0")
  return b
