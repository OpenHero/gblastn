#!/bin/sh
# Native Linux build; no GPU or host NVIDIA driver is needed for compilation.
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
: "${CC:=gcc-13}" "${CXX:=g++-13}" "${CUDA_HOME:=/usr/local/cuda}"
: "${CUDA_ARCH:=110}" "${PREFIX:=/opt/gblastn}" "${JOBS:=4}"
: "${BUILD_ROOT:=$repo/.build/toolkit}" "${NVCC:=$CUDA_HOME/bin/nvcc}"
: "${CFLAGS:=-O2 -fcommon}" "${CXXFLAGS:=-O2 -std=gnu++14 -Wno-deprecated-declarations}"
case $CUDA_ARCH in *[!0-9]*|'') echo 'CUDA_ARCH must be a numeric compute capability (Thor: 110)' >&2; exit 2;; esac
case $JOBS in *[!0-9]*|0|'') echo 'JOBS must be a positive integer' >&2; exit 2;; esac
[ "$(uname -s)" = Linux ] || { echo 'Build inside the Linux ARM64 container (or natively on Linux).' >&2; exit 2; }
command -v "$CC" >/dev/null
command -v "$CXX" >/dev/null
[ -x "$NVCC" ] || { echo "nvcc not found: $NVCC" >&2; exit 2; }
if [ -z "${CUDA_LIBDIR:-}" ]; then
    for d in "$CUDA_HOME/targets/sbsa-linux/lib" "$CUDA_HOME/targets/$(uname -m)-linux/lib" "$CUDA_HOME/lib64" "$CUDA_HOME/lib"; do
        if [ -f "$d/libcudart.so" ]; then CUDA_LIBDIR=$d; break; fi
    done
fi
: "${CUDA_LIBDIR:?Set CUDA_LIBDIR to the directory containing libcudart.so}"
export CC CXX CUDA_HOME CUDA_LIBDIR CUDA_ARCH NVCC CFLAGS CXXFLAGS
# GNU make propagates its jobserver through MAKEFLAGS. The 2013 recipes also
# pass MFLAGS explicitly; clearing that duplicate prevents -j from spawning a
# fresh jobserver at every recursive level and exceeding JOBS.
export MAKEFLAGS="${MAKEFLAGS:-} -j$JOBS MFLAGS="
export CPPFLAGS="${CPPFLAGS:-} -I$CUDA_HOME/include"
export LDFLAGS="${LDFLAGS:-} -L$CUDA_LIBDIR"
mkdir -p "$BUILD_ROOT/configure"
cd "$BUILD_ROOT/configure"
"$repo/c++/configure" --with-build-root="$BUILD_ROOT" --without-suffix \
    --prefix="$PREFIX" --without-debug --with-mt --with-autodep --without-flat-makefile \
    --without-boost --without-sybase --without-fastcgi --without-sssdb \
    --without-sss --without-geo --without-sp --without-orbacus \
    --without-openssl --without-gnutls --without-libxml --without-sqlite3 \
    --without-bdb --without-python --without-perl \
    --with-projects="$repo/c++/scripts/projects/blast/project.lst" "$@"
# Bootstrap locally after configure (which clears MAKEFLAGS internally). This
# preserves JOBS and propagates failures instead of silently omitting the graph.
"$repo/c++/scripts/common/impl/create_flat_makefile.sh" "$BUILD_ROOT" \
    -s "$repo/c++" -p "$repo/c++/scripts/projects/blast/project.lst" -b
# The flat dependency graph schedules the CUDA library before applications.
make -C "$BUILD_ROOT/build" -f Makefile.flat -j "$JOBS" \
    blastn.exe.real makeblastdb.exe.real blastdbcmd.exe.real
mkdir -p "$PREFIX/bin" "$PREFIX/share/gblastn"
for exe in blastn makeblastdb blastdbcmd; do
    install -m 755 "$BUILD_ROOT/bin/$exe" "$PREFIX/bin/$exe"
done
{
    printf 'NCBI baseline: 2.2.28+\nCUDA target: sm_%s + compute_%s PTX\n' "$CUDA_ARCH" "$CUDA_ARCH"
    "$CC" --version | head -n 1
    "$CXX" --version | head -n 1
    "$NVCC" --version
    uname -m
} > "$PREFIX/share/gblastn/build-info.txt"
# Fail packaging if any shared library is missing; libcudart stays in CUDA.
for exe in blastn makeblastdb blastdbcmd; do
    ldd "$PREFIX/bin/$exe" > "$PREFIX/share/gblastn/$exe.ldd.txt"
    if grep -q 'not found' "$PREFIX/share/gblastn/$exe.ldd.txt"; then
        cat "$PREFIX/share/gblastn/$exe.ldd.txt" >&2; exit 1
    fi
done
