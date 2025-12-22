#!/bin/bash
set -e

# Get project root directory (3 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORK_DIR="/tmp/temp_ffmpeg"
INSTALL_DIR="$PROJECT_DIR/bin"
FFMPEG_VERSION="7.1"

echo "Project directory: $PROJECT_DIR"
echo "Work directory: $WORK_DIR"
echo "Install directory: $INSTALL_DIR"

# Check if ffmpeg already exists
if [ -f "$INSTALL_DIR/ffmpeg" ]; then
    echo "ffmpeg already exists at $INSTALL_DIR/ffmpeg"
    ls -lh "$INSTALL_DIR/ffmpeg"
    exit 0
fi

echo "Creating work directory..."
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
mkdir -p "$INSTALL_DIR"
cd "$WORK_DIR"

# Install build dependencies
echo "Installing build dependencies..."
apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    yasm \
    nasm \
    pkg-config \
    zlib1g-dev \
    curl \
    ca-certificates

# Download and build x264
echo "Downloading and building x264..."
git clone --depth 1 https://code.videolan.org/videolan/x264.git
cd x264
./configure \
    --prefix="$WORK_DIR/deps" \
    --enable-static \
    --disable-shared \
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
    \
    --disable-avdevice \
    --disable-postproc \
    \
    --disable-encoders \
    --enable-encoder=libx264 \
    --enable-encoder=png \
    \
    --disable-decoders \
    --enable-decoder=png \
    \
    --disable-muxers \
    --enable-muxer=mp4 \
    --enable-muxer=image2 \
    \
    --disable-demuxers \
    --enable-demuxer=image2 \
    \
    --disable-parsers \
    --enable-parser=png \
    \
    --disable-protocols \
    --enable-protocol=file \
    \
    --disable-filters \
    --enable-filter=scale \
    --enable-filter=format \
    --enable-filter=null \
    \
    --disable-bsfs \
    --disable-indevs \
    --disable-outdevs \
    --extra-cflags="-I$WORK_DIR/deps/include" \
    --extra-ldflags="-L$WORK_DIR/deps/lib"

# Build
echo "Building ffmpeg..."
make -j$(nproc)

# Copy and strip the binary
echo "Installing ffmpeg to $INSTALL_DIR..."
cp ffmpeg "$INSTALL_DIR/ffmpeg"
strip "$INSTALL_DIR/ffmpeg"

# Clean up
echo "Cleaning up..."
cd /
rm -rf "$WORK_DIR"

# Show result
echo ""
echo "===== SUCCESS ====="
ls -lh "$INSTALL_DIR/ffmpeg"
