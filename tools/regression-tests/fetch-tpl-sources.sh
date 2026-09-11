#!/bin/bash
# Provide the third party libraries that LAMMPS cannot download by itself in a
# restricted environment, and build the ones that have to be installed before
# cmake can find them.
#
# Several packages fetch a release tarball from github.com.  Where that host is
# reachable nothing here is needed and the script does nothing.  Where only the
# git protocol gets through - which is a common shape of egress policy, since
# "git clone https://github.com/..." and a plain GET of a release asset are
# treated differently - the sources are cloned at their pinned tag instead and
# repacked into a local tarball.  LAMMPS then takes it through the <PREFIX>_URL
# and <PREFIX>_SHA256 cache variables that SetDownloadSettings() defines, so no
# cmake code has to change:
#
#   cmake ... -D SCAFACOS_URL=file://${TPL}/scafacos-1.0.4-bootstrapped.tar.gz \
#             -D SCAFACOS_SHA256=$(sha256sum ...)
#
# ADIOS2 and xtb are different: their LAMMPS packages look for an installed
# library rather than downloading anything, so those are built and installed
# into ${TPL}/install and found through CMAKE_PREFIX_PATH and PKG_CONFIG_PATH.
set -euo pipefail

TPL=${TPL:-${PWD}/tpl}
SRC=${TPL}/src
PREFIX=${TPL}/install
JOBS=${JOBS:-4}
mkdir -p "${SRC}" "${PREFIX}"

have() { command -v "$1" > /dev/null 2>&1; }

# $1 repo url, $2 tag ("-" for the default branch), $3 directory name
clone() {
    local url=$1 tag=$2 dir=$3
    [ -d "${SRC}/${dir}" ] && { echo "${dir}: already cloned"; return 0; }
    if [ "${tag}" = "-" ]; then
        git clone --quiet --depth 1 "${url}" "${SRC}/${dir}"
    else
        git clone --quiet --depth 1 --branch "${tag}" "${url}" "${SRC}/${dir}"
    fi
}

pack_autotools() {
    local dir=$1 name=$2 out="${TPL}/${name}.tar.gz"
    [ -f "${out}" ] && { echo "${name}: already packed"; return 0; }
    pushd "${SRC}/${dir}" > /dev/null
    # the release tarballs of these projects ship a generated "configure"; a
    # checkout of the tag does not, and LAMMPS runs <SOURCE_DIR>/configure
    if [ ! -x configure ]; then
        if [ -x ./bootstrap ]; then ./bootstrap; else autoreconf -fi; fi
    fi
    rm -rf .git
    popd > /dev/null
    tar czf "${out}" -C "${SRC}" --transform "s,^${dir},${name}," "${dir}"
    echo "${name}: packed, sha256 $(sha256sum "${out}" | cut -d' ' -f1)"
}

case "${1:-all}" in
  scafacos|all)
    clone https://github.com/scafacos/scafacos.git v1.0.4 scafacos
    pack_autotools scafacos scafacos-1.0.4-bootstrapped
    ;;&
  mbx|all)
    clone https://github.com/paesanilab/MBX.git v1.4.0 mbx
    pack_autotools mbx mbx-1.4.0-bootstrapped
    ;;&
  adios2|all)
    # LAMMPS refuses an ADIOS2 built without MPI when BUILD_MPI is on
    clone https://github.com/ornladios/ADIOS2.git - adios2
    if [ ! -f "${PREFIX}/lib/cmake/adios2/adios2-config.cmake" ] && \
       [ ! -f "${PREFIX}/lib64/cmake/adios2/adios2-config.cmake" ]; then
        cmake -S "${SRC}/adios2" -B "${SRC}/adios2-build" -G Ninja \
              -D CMAKE_INSTALL_PREFIX="${PREFIX}" -D CMAKE_BUILD_TYPE=Release \
              -D ADIOS2_USE_MPI=ON -D ADIOS2_USE_Fortran=OFF \
              -D ADIOS2_USE_Python=OFF -D ADIOS2_BUILD_EXAMPLES=OFF \
              -D BUILD_TESTING=OFF -D ADIOS2_USE_HDF5=OFF
        cmake --build "${SRC}/adios2-build" -j "${JOBS}"
        cmake --install "${SRC}/adios2-build"
    fi
    echo "adios2: installed in ${PREFIX}"
    ;;&
  xtb|all)
    have meson || { echo "xtb: needs meson, skipped"; exit 0; }
    clone https://github.com/grimme-lab/mctc-lib.git - mctc-lib
    clone https://github.com/grimme-lab/xtb.git - xtb
    for p in mctc-lib xtb; do
        if [ ! -d "${SRC}/${p}-build" ]; then
            meson setup "${SRC}/${p}-build" "${SRC}/${p}" \
                  --prefix="${PREFIX}" --libdir=lib --buildtype=release
        fi
        meson install -C "${SRC}/${p}-build"
    done
    echo "xtb: installed in ${PREFIX}"
    ;;
esac

cat <<EOM

Pass these to the LAMMPS cmake invocation:
  -D CMAKE_PREFIX_PATH=${PREFIX}
  -D SCAFACOS_URL=file://${TPL}/scafacos-1.0.4-bootstrapped.tar.gz
  -D SCAFACOS_SHA256=<sha256 printed above>
  -D MBXLIB_URL=file://${TPL}/mbx-1.4.0-bootstrapped.tar.gz
  -D MBXLIB_SHA256=<sha256 printed above>
and export PKG_CONFIG_PATH=${PREFIX}/lib/pkgconfig:\${PKG_CONFIG_PATH}
EOM
