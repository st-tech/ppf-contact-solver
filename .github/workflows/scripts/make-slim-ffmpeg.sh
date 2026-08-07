#!/bin/bash
set -e

# Get project root directory (3 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORK_DIR="/tmp/temp_ffmpeg"
INSTALL_DIR="$PROJECT_DIR/bin"

# Load the URL/tag manifest. It is shared with the Windows build rather than
# copied, so the two platforms cannot come to pin different ffmpeg revisions.
MANIFEST="$PROJECT_DIR/build-win-native/scripts/downloads.txt"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Manifest not found: $MANIFEST" >&2
    exit 1
fi
set -a
# shellcheck disable=SC1090
. "$MANIFEST"
set +a

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
    git \
    ca-certificates

# Download and build x264
echo "Downloading and building x264..."
git clone --depth 1 "${URL_X264_GIT}"
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

# Clone ffmpeg source at its release tag. --branch takes a tag, and git fails
# non-zero when it names nothing, so a retired tag aborts here under set -e
# rather than configuring an unexpected tree.
echo "Cloning ffmpeg ${FFMPEG_VERSION} (${FFMPEG_TAG})..."
git clone --depth 1 --branch "${FFMPEG_TAG}" "${URL_FFMPEG_GIT}" "ffmpeg-${FFMPEG_VERSION}"
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
