#!/bin/sh
# Install the OpenKIM models that the examples in examples/kim use.
#
# "kim-api-collections-management install system <model-id>" normally resolves a
# model and its driver through query.openkim.org, but that host is not always
# reachable while https://openkim.org/download/<id>.txz is.  So each item is
# fetched by its download URL and installed from the unpacked directory, and the
# model driver named in the model's kimspec.edn is installed first.
#
# Usage: install-kim-models.sh <kim-api-prefix> [<model-id> ...]
set -eu

PREFIX=${1:?usage: install-kim-models.sh <kim-api-prefix> [<model-id> ...]}
shift

MODELS=${*:-"\
EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005 \
EAM_Dynamo_MendelevAckland_2007v3_Zr__MO_004835508849_000 \
EAM_Dynamo_WineyKubotaGupta_2010_Al__MO_149316865608_005 \
LJ_Shifted_Bernardes_1958MedCutoff_Ar__MO_126566794224_004 \
SW_StillingerWeber_1985_Si__MO_405512056662_005 \
Sim_LAMMPS_ReaxFF_StrachanVanDuinChakraborty_2003_CHNO__SM_107643900657_000"}

MANAGE="${PREFIX}/bin/kim-api-collections-management"
# the items are built with CMake and look for the KIM-API package config, which
# lives under the prefix rather than in a system location
CMAKE_PREFIX_PATH="${PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export CMAKE_PREFIX_PATH
WORK=$(mktemp -d)
trap 'rm -rf "${WORK}"' EXIT

fetch_and_unpack() {
    id=$1
    if [ -d "${WORK}/${id}" ]; then return 0; fi
    echo "--- fetching ${id}"
    curl -fsSL -o "${WORK}/${id}.txz" "https://openkim.org/download/${id}.txz"
    tar -x -J -f "${WORK}/${id}.txz" -C "${WORK}"
}

for model in ${MODELS}; do
    fetch_and_unpack "${model}"
    # a portable model names its driver in kimspec.edn; a simulator model has none
    driver=$(sed -n 's/.*"model-driver"[[:space:]]*"\([^"]*\)".*/\1/p' \
                 "${WORK}/${model}/kimspec.edn" 2>/dev/null || true)
    if [ -n "${driver}" ]; then
        fetch_and_unpack "${driver}"
        echo "--- installing driver ${driver}"
        "${MANAGE}" install --force system "${WORK}/${driver}" > /dev/null
    fi
    echo "--- installing ${model}"
    "${MANAGE}" install --force system "${WORK}/${model}" > /dev/null
done

"${MANAGE}" list
