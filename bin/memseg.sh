#!/bin/bash


echo "Membrane Segmentation : Segment membrane in cryoET tomogram"
echo " Note: Run memseg.sh --help for comamnd line help"

SOURCE="${BASH_SOURCE[0]}"
while [ -h "${SOURCE}" ]; do
  SCRIPTDIR="$(cd -P "$(dirname "${SOURCE}")" >/dev/null && pwd)"
  SOURCE="$(readlink "${SOURCE}")"
  [[ ${SOURCE} != /* ]] && SOURCE="${SCRIPTDIR}/${SOURCE}"
done
PROOT="$(cd -P "$(dirname "${SOURCE}")" >/dev/null && pwd)"

PYTHON="$(which python)"

env -i ${PYTHON} -s ${PROOT}/../mpicker_gui/Mpicker_memseg.py $*


exit 0


