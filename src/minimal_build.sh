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

echo Building with 'bench = no, minimal = yes'
make -j profile-build ARCH=$ARCH COMP=clang bench=no minimal=yes > /dev/null

make strip
print_size stockfish

if [ -f stockfish.zopfli.gz ]; then
  echo "before: $(print_size stockfish.zopfli.gz)"
fi
if [ -f stockfish.xz ]; then
  echo "before: $(print_size stockfish.xz)"
  rm stockfish.xz
fi
if [ -f stockfish.bz2 ]; then
  echo "before: $(print_size stockfish.bz2)"
  rm stockfish.bz2
fi
if [ -f stockfish.7z ]; then
  echo "before: $(print_size stockfish.7z)"
fi
if [ -f stockfish.zip ]; then
  echo "before: $(print_size stockfish.zip)"
fi
echo

zopfli --i100 -c stockfish > stockfish.zopfli.gz
xz -k -9 stockfish
bzip2 -k -9 stockfish
# p7zip -k -f stockfish > /dev/null
7z a -t7z -mx=9 stockfish.7z ./stockfish > /dev/null
zip -9 stockfish.zip stockfish > /dev/null

echo "after:"
print_size stockfish.zopfli.gz
print_size stockfish.xz
print_size stockfish.bz2
print_size stockfish.7z
print_size stockfish.zip

# nodes_searched=$(./stockfish bench 2>&1 | grep "Nodes searched" | grep -oE "\d+")
# echo bench: $nodes_searched

# echo
# echo "full submission size:"
# rm -f submission.7z
# 7z a -t7z -mx=9 submission.7z ./main.py ./stockfish
 #print_size submission.7z

renamed=stockfish-$(git rev-parse --short=6 HEAD)-profile-$ARCH
echo
echo $renamed
mv stockfish $renamed
