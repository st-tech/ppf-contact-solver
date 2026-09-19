#!/bin/bash
# File: make-slim-ffmpeg.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Build slim ffmpeg for Windows using MSYS2/MinGW-w64

# Get script directory BEFORE sourcing profile (which changes cwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The MSYS2 environment that targets this host, chosen by make-slim-ffmpeg.bat:
# MINGW64 on x64 and CLANGARM64 on ARM64. Run directly, it is MINGW64.
export MSYSTEM="${PPF_MSYSTEM:-MINGW64}"
source /etc/profile || true

# CLANGARM64 carries clang and the LLVM binutils under their own names and no
# gcc, which is the compiler ffmpeg's configure looks for by default, so the
# compiler and strip are named for it.
case "$MSYSTEM" in
    CLANGARM64)
        export CC=clang CXX=clang++
        # THE ARCHITECTURE AND OS ARE NAMED, NOT DETECTED. MSYS2 itself is an x64
        # program, run emulated on ARM64 Windows, so `uname -m` answers x86_64,
        # and ffmpeg's configure takes its arch default from that: on the
        # windows-11-arm runner (Build Windows 34991554786) configure stopped on
        # "nasm/yasm not found or too old", demanding the x86 assembler, after
        # x264 had built its aarch64 assembly with this same clang. `uname -s`
        # here names the CLANGARM64 environment, which configure's mingw32* and
        # mingw64* patterns do not match, so the OS is named too. With no
        # --cross-prefix this is not a cross compile to configure, and it takes
        # the host compiler from --cc.
        FFMPEG_CC_FLAGS=(--cc=clang --cxx=clang++ --arch=aarch64 --target-os=mingw32)
        STRIP=llvm-strip
        ;;
    MINGW64)
        FFMPEG_CC_FLAGS=()
        STRIP=strip
        ;;
    *)
        echo "ERROR: make-slim-ffmpeg.sh builds in MINGW64 or CLANGARM64, and MSYSTEM is $MSYSTEM" >&2
        exit 1
        ;;
esac

set -e
FFMPEG_DIR="$SCRIPT_DIR/ffmpeg"
WORK_DIR="$SCRIPT_DIR/temp_ffmpeg"

# Load the URL/FILE/tag manifest (single source of truth, scripts/downloads.txt)
MANIFEST="$SCRIPT_DIR/scripts/downloads.txt"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Manifest not found: $MANIFEST" >&2
    exit 1
fi
set -a
# shellcheck disable=SC1090
. "$MANIFEST"
set +a

echo "Script directory: $SCRIPT_DIR"
echo "FFmpeg directory: $FFMPEG_DIR"
echo "Work directory: $WORK_DIR"

# Check if ffmpeg already exists
if [ -f "$FFMPEG_DIR/ffmpeg.exe" ]; then
    echo "ffmpeg already exists at $FFMPEG_DIR/ffmpeg.exe"
    ls -lh "$FFMPEG_DIR/ffmpeg.exe"
    exit 0
fi

echo "Creating work directory..."
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
mkdir -p "$FFMPEG_DIR"
cd "$WORK_DIR"

# Download and build x264
echo "Downloading and building x264..."
git clone --depth 1 "${URL_X264_GIT}"
cd x264
./configure \
    --prefix="$WORK_DIR/deps" \
    --enable-static \
    --disable-cli \
    --disable-opencl \
    --disable-avs \
    --disable-swscale \
    --disable-lavf \
    --disable-ffms \
    --disable-gpac \
    --disable-lsmash
make -j$(nproc)
make install
cd ..

# Clone ffmpeg source at its release tag. --branch takes a tag, and git fails
# non-zero when it names nothing, so a retired tag aborts here under set -e
# rather than configuring an unexpected tree.
echo "Cloning ffmpeg ${FFMPEG_VERSION} (${FFMPEG_TAG})..."
git clone --depth 1 --branch "${FFMPEG_TAG}" "${URL_FFMPEG_GIT}" "ffmpeg-${FFMPEG_VERSION}"
cd "ffmpeg-${FFMPEG_VERSION}"

# Configure with minimal options for PNG to MP4
echo "Configuring ffmpeg with minimal options..."
PKG_CONFIG_PATH="$WORK_DIR/deps/lib/pkgconfig:$PKG_CONFIG_PATH" ./configure \
    ${FFMPEG_CC_FLAGS[@]+"${FFMPEG_CC_FLAGS[@]}"} \
    --prefix="$WORK_DIR/output" \
    --enable-gpl \
    --enable-libx264 \
    --enable-zlib \
    --enable-static \
    --disable-shared \
    --disable-doc \
    --disable-htmlpages \
    --disable-manpages \
    --disable-podpages \
    --disable-txtpages \
    --disable-network \
    --disable-autodetect \
    --disable-iconv \
    --disable-debug \
    --disable-ffplay \
    --disable-ffprobe \
    --disable-avdevice \
    --disable-postproc \
    --disable-encoders \
    --enable-encoder=libx264 \
    --enable-encoder=png \
    --disable-decoders \
    --enable-decoder=png \
    --disable-muxers \
    --enable-muxer=mp4 \
    --enable-muxer=image2 \
    --disable-demuxers \
    --enable-demuxer=image2 \
    --disable-parsers \
    --enable-parser=png \
    --disable-protocols \
    --enable-protocol=file \
    --disable-filters \
    --enable-filter=scale \
    --enable-filter=format \
    --enable-filter=null \
    --disable-bsfs \
    --disable-indevs \
    --disable-outdevs \
    --extra-cflags="-I$WORK_DIR/deps/include" \
    --extra-ldflags="-L$WORK_DIR/deps/lib -static"

# Build
echo "Building ffmpeg..."
make -j$(nproc)

# Copy and strip the binary
echo "Installing ffmpeg to $FFMPEG_DIR..."
cp ffmpeg.exe "$FFMPEG_DIR/ffmpeg.exe"
"$STRIP" "$FFMPEG_DIR/ffmpeg.exe"

# Clean up
echo "Cleaning up..."
cd /
rm -rf "$WORK_DIR"

# Show result
echo ""
echo "===== SUCCESS ====="
ls -lh "$FFMPEG_DIR/ffmpeg.exe"
