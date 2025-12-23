#!/bin/bash
# File: make-slim-ffmpeg.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Build slim ffmpeg for Windows using MSYS2/MinGW-w64

# Get script directory BEFORE sourcing profile (which changes cwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure we're in MinGW64 environment
export MSYSTEM=MINGW64
source /etc/profile || true

set -e
FFMPEG_DIR="$SCRIPT_DIR/ffmpeg"
WORK_DIR="$SCRIPT_DIR/temp_ffmpeg"
FFMPEG_VERSION="7.1"

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
git clone --depth 1 https://code.videolan.org/videolan/x264.git
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

# Download ffmpeg source
echo "Downloading ffmpeg ${FFMPEG_VERSION}..."
curl -L "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" -o ffmpeg.tar.xz
tar xf ffmpeg.tar.xz
cd "ffmpeg-${FFMPEG_VERSION}"

# Configure with minimal options for PNG to MP4
echo "Configuring ffmpeg with minimal options..."
PKG_CONFIG_PATH="$WORK_DIR/deps/lib/pkgconfig:$PKG_CONFIG_PATH" ./configure \
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
strip "$FFMPEG_DIR/ffmpeg.exe"

# Clean up
echo "Cleaning up..."
cd /
rm -rf "$WORK_DIR"

# Show result
echo ""
echo "===== SUCCESS ====="
ls -lh "$FFMPEG_DIR/ffmpeg.exe"
