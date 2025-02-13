# #!/bin/bash
set -eu -o pipefail

make clean

if [[ "$(uname)" == "Darwin" ]]; then
  ARCH=x86-64-modern
else
  ARCH=x86-64-bmi2
fi

function print_size() {
  ls -lt $1 | awk '{printf "%-10s %s\n", $5, $9}'
}

echo Building with 'bench = no'
make -j profile-build ARCH=$ARCH COMP=clang bench=no minimal=yes > /dev/null

make strip
print_size stockfish

rm -f stockfish.7z e.7z
mv stockfish e
7z a -t7z -mx=9 e.7z ./e > /dev/null
print_size e.7z

rm -f submission.tar
tar -cvf submission.tar e.7z main.py
print_size submission.tar

rm -f submission.tar.gz
zopfli --i1000 submission.tar
print_size submission.tar.gz
