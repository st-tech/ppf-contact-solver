#!/bin/bash
# File: .github/workflows/scripts/make-slim-ffmpeg.sh
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Builds the slim ffmpeg (PNG frames to an x264 MP4, nothing else) on Linux.
# Two callers: the Dockerfile's builder stage, and build-linux-native/warmup.sh,
# which runs it without root on a host that may have no network.
#
# IT INSTALLS NOTHING. The build tools (a C compiler, make, nasm, pkg-config,
# git, and curl when sources are fetched) are the caller's to provide, so the
# script runs unprivileged. The Dockerfile installs them with apt before calling.
#
# Settings, all optional, read from the environment:
#   FFMPEG_INSTALL_DIR  where the binary goes (default: <project>/bin)
#   FFMPEG_WORK_DIR     scratch directory, removed afterwards
#                       (default: /tmp/temp_ffmpeg)
#   FFMPEG_SOURCE_DIR   a directory holding pre-fetched sources, for a host with
#                       no network: the x264 clone as `x264`, the ffmpeg clone
#                       as `ffmpeg-<FFMPEG_TAG>`, and the zlib archive under its
#                       manifest name. Unset means clone and download them.
#
# WHAT THE BINARY IS BUILT FROM IS PINNED AND CHECKED, NOT ASSUMED. ffmpeg is a
# tag, x264 a commit, zlib an archive with a checksum, all from the manifest
# shared with the Windows build. Pre-fetched sources are checked against the
# same pins as fetched ones. The build is --enable-gpl through x264, so a
# distribution shipping this binary has to name its exact sources, which the
# script records beside the binary in ffmpeg.sources.txt.
#
# ZLIB IS LINKED STATICALLY. The PNG decoder needs it, and a binary that loaded
# the host's shared libz would carry a dependency that a self-contained
# distribution cannot assume on the machine it lands on. The check at the end
# refuses a binary that needs libz or libx264 at run time.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORK_DIR="${FFMPEG_WORK_DIR:-/tmp/temp_ffmpeg}"
INSTALL_DIR="${FFMPEG_INSTALL_DIR:-$PROJECT_DIR/bin}"
SOURCE_DIR="${FFMPEG_SOURCE_DIR:-}"

die() {
    printf 'ERROR: %s\n' "$1" >&2
    shift
    for line in "$@"; do
        printf '       %s\n' "$line" >&2
    done
    exit 1
}

# Load the URL/tag manifest. It is shared with the Windows build rather than
# copied, so the platforms cannot come to pin different ffmpeg revisions.
MANIFEST="$PROJECT_DIR/build-win-native/scripts/downloads.txt"
[ -f "$MANIFEST" ] || die "manifest not found: $MANIFEST"
set -a
# shellcheck disable=SC1090
. "$MANIFEST"
set +a
for key in URL_FFMPEG_GIT FFMPEG_TAG URL_X264_GIT X264_COMMIT URL_ZLIB FILE_ZLIB SHA256_ZLIB; do
    [ -n "${!key:-}" ] || die "$MANIFEST does not set $key"
done

echo "Project directory: $PROJECT_DIR"
echo "Work directory:    $WORK_DIR"
echo "Install directory: $INSTALL_DIR"
echo "Source directory:  ${SOURCE_DIR:-<none, sources are fetched>}"

if [ -f "$INSTALL_DIR/ffmpeg" ]; then
    echo "ffmpeg already exists at $INSTALL_DIR/ffmpeg"
    ls -lh "$INSTALL_DIR/ffmpeg"
    exit 0
fi

# nasm assembles x264's x86 code and nothing else: on aarch64 x264's assembly goes
# through the C compiler, so the assembler is asked for only where it is used.
# The loader is the one run-time dependency whose name differs by architecture.
case "$(uname -m)" in
    x86_64)
        needed_tools="cc make nasm pkg-config git sha256sum tar readelf strip"
        LOADER_PATTERN='ld-linux-x86-64.so.*'
        ;;
    aarch64)
        needed_tools="cc make pkg-config git sha256sum tar readelf strip"
        LOADER_PATTERN='ld-linux-aarch64.so.*'
        ;;
    *)
        die "this script builds on x86_64 and aarch64, and uname -m reports $(uname -m)"
        ;;
esac
[ -n "$SOURCE_DIR" ] || needed_tools="$needed_tools curl"
for tool in $needed_tools; do
    command -v "$tool" >/dev/null 2>&1 || die \
        "required tool not found on PATH: $tool" \
        "This script installs nothing. Install the build tools first, for" \
        "example: apt-get install build-essential nasm pkg-config curl git"
done

echo "Creating work directory..."
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR/deps" "$INSTALL_DIR"
cd "$WORK_DIR"

# zlib, static, from an archive checked against its manifest hash.
echo "Building zlib from $FILE_ZLIB..."
if [ -n "$SOURCE_DIR" ]; then
    [ -f "$SOURCE_DIR/$FILE_ZLIB" ] || die "no $FILE_ZLIB in $SOURCE_DIR"
    cp "$SOURCE_DIR/$FILE_ZLIB" "$WORK_DIR/"
else
    curl -fL --retry 3 --retry-delay 2 --connect-timeout 15 -o "$WORK_DIR/$FILE_ZLIB" "$URL_ZLIB"
fi
zlib_sum="$(sha256sum "$WORK_DIR/$FILE_ZLIB" | cut -d' ' -f1)"
[ "$zlib_sum" = "$SHA256_ZLIB" ] || die \
    "checksum mismatch on $FILE_ZLIB" \
    "expected $SHA256_ZLIB" \
    "measured $zlib_sum"
mkdir -p zlib
tar -xzf "$FILE_ZLIB" -C zlib --strip-components=1
(
    cd zlib
    ./configure --static --prefix="$WORK_DIR/deps"
    make -j"$(nproc)"
    make install
)

# x264 at its pinned commit.
echo "Building x264 at $X264_COMMIT..."
if [ -n "$SOURCE_DIR" ]; then
    [ -d "$SOURCE_DIR/x264/.git" ] || die "no x264 clone at $SOURCE_DIR/x264"
    cp -a "$SOURCE_DIR/x264" x264
else
    git clone -q "$URL_X264_GIT" x264
fi
git -C x264 -c advice.detachedHead=false checkout -q "$X264_COMMIT" || die \
    "x264 has no commit $X264_COMMIT" \
    "The pin in $MANIFEST names a commit this clone does not carry."
[ "$(git -C x264 rev-parse HEAD)" = "$X264_COMMIT" ] || die "x264 is not at $X264_COMMIT"
(
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
    make -j"$(nproc)"
    make install
)

# ffmpeg at its release tag. --branch takes a tag, and git fails non-zero when
# it names nothing, so a retired tag aborts here rather than configuring an
# unexpected tree. A pre-fetched clone is held to the same tag.
echo "Preparing ffmpeg ${FFMPEG_TAG}..."
FFMPEG_TREE="ffmpeg-${FFMPEG_TAG}"
if [ -n "$SOURCE_DIR" ]; then
    [ -d "$SOURCE_DIR/$FFMPEG_TREE/.git" ] || die "no ffmpeg clone at $SOURCE_DIR/$FFMPEG_TREE"
    cp -a "$SOURCE_DIR/$FFMPEG_TREE" "$FFMPEG_TREE"
else
    git clone -q --depth 1 --branch "${FFMPEG_TAG}" "${URL_FFMPEG_GIT}" "$FFMPEG_TREE"
fi
ffmpeg_commit="$(git -C "$FFMPEG_TREE" rev-parse HEAD)"
tag_commit="$(git -C "$FFMPEG_TREE" rev-parse "${FFMPEG_TAG}^{commit}" 2>/dev/null)" || die \
    "the ffmpeg clone carries no tag ${FFMPEG_TAG}"
[ "$ffmpeg_commit" = "$tag_commit" ] || die \
    "the ffmpeg clone is at $ffmpeg_commit, and ${FFMPEG_TAG} is $tag_commit"
cd "$FFMPEG_TREE"

# Configure with minimal options for PNG to MP4.
#
# NO PATH OF THIS MACHINE GOES ON THE CONFIGURE LINE, because ffmpeg compiles
# that line into the binary, which prints it for `ffmpeg -version`, and turns
# --prefix into the data directory it searches at run time. The static
# dependencies are found through the compiler's and pkg-config's own environment
# variables, which configure does not record, and --prefix stays at its default
# because the binary is copied out rather than installed. The check after the
# build refuses a binary that still names the work directory.
echo "Configuring ffmpeg with minimal options..."
export C_INCLUDE_PATH="$WORK_DIR/deps/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export LIBRARY_PATH="$WORK_DIR/deps/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export PKG_CONFIG_PATH="$WORK_DIR/deps/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
./configure \
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
    --disable-outdevs

echo "Building ffmpeg..."
make -j"$(nproc)"

# THE RUN-TIME DEPENDENCIES ARE CHECKED ON THE BINARY, not inferred from the
# configure line: a shared libz found ahead of the static one would link
# cleanly and then need the host's libz wherever the binary goes.
needed="$(readelf -d ffmpeg | awk '/\(NEEDED\)/ { gsub(/[\[\]]/, "", $NF); print $NF }')"
printf 'ffmpeg needs: %s\n' "$(printf '%s' "$needed" | tr '\n' ' ')"
for lib in $needed; do
    case "$lib" in
        # $LOADER_PATTERN is unquoted on purpose: a case pattern taken from a
        # variable is matched as a pattern only when it is not quoted.
        libc.so.* | libm.so.* | libpthread.so.* | libdl.so.* | librt.so.* | $LOADER_PATTERN) ;;
        *) die "ffmpeg needs $lib at run time" \
               "Only the C library family is expected. libz and libx264 are linked" \
               "statically so the binary needs nothing a distribution cannot assume." ;;
    esac
done

echo "Installing ffmpeg to $INSTALL_DIR..."
cp ffmpeg "$INSTALL_DIR/ffmpeg"
strip "$INSTALL_DIR/ffmpeg"

# NO PATH OF THE BUILDING MACHINE IS LEFT IN THE BINARY, checked on the stripped
# copy that ships. Anything naming the work directory would be printed by
# `ffmpeg -version` wherever the binary goes, or searched at run time.
if grep -qaF "$WORK_DIR" "$INSTALL_DIR/ffmpeg"; then
    die "ffmpeg names its work directory $WORK_DIR" \
        "$(grep -aoF "$WORK_DIR" "$INSTALL_DIR/ffmpeg" | wc -l) occurrences; the configure line or a" \
        "dependency's build recorded a path of this machine."
fi
# The license texts of the three sources compiled into the binary, taken from
# those sources rather than restated, so a distribution shipping the binary can
# ship them beside it.
mkdir -p "$INSTALL_DIR/licenses"
cp COPYING.GPLv2 "$INSTALL_DIR/licenses/ffmpeg-COPYING.GPLv2"
cp LICENSE.md "$INSTALL_DIR/licenses/ffmpeg-LICENSE.md"
cp "$WORK_DIR/x264/COPYING" "$INSTALL_DIR/licenses/x264-COPYING"
cp "$WORK_DIR/zlib/LICENSE" "$INSTALL_DIR/licenses/zlib-LICENSE"
cat > "$INSTALL_DIR/ffmpeg.sources.txt" <<EOF
ffmpeg  ${FFMPEG_TAG} (${ffmpeg_commit})  ${URL_FFMPEG_GIT}
x264    ${X264_COMMIT}  ${URL_X264_GIT}
zlib    ${FILE_ZLIB} (sha256 ${SHA256_ZLIB})  ${URL_ZLIB}
Configured with --enable-gpl --enable-libx264, so the binary is distributed
under the GNU General Public License, version 2 or later.
EOF

echo "Cleaning up..."
cd /
rm -rf "$WORK_DIR"

echo ""
echo "===== SUCCESS ====="
ls -lh "$INSTALL_DIR/ffmpeg"
