#!/bin/bash


echo "A wrapper for mpicker python scripts, Mpicker_xxx.py"
echo " Example: mpicker_python.sh Mpicker_particles.py --help"
echo " Note: Run mpicker_python.sh to list scripts"

SOURCE="${BASH_SOURCE[0]}"
while [ -h "${SOURCE}" ]; do
  SCRIPTDIR="$(cd -P "$(dirname "${SOURCE}")" >/dev/null && pwd)"
  SOURCE="$(readlink "${SOURCE}")"
  [[ ${SOURCE} != /* ]] && SOURCE="${SCRIPTDIR}/${SOURCE}"
done
PROOT="$(cd -P "$(dirname "${SOURCE}")" >/dev/null && pwd)"

PYTHON="$(which python)"

if [ -f "${PROOT}/../mpicker_gui/$1" ]; then
  env -i ${PYTHON} -s ${PROOT}/../mpicker_gui/$*
else
  echo "mpicker scripts:"
  basename -a  ${PROOT}/../mpicker_gui/Mpicker_*.py
fi


exit 0



