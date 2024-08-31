#!/bin/bash

echo "You can edit config file after you change the position of data."

case $1 in
"inputboundary")
    if [ $# -ne 1 ]; then
    echo "Usage: mpicker_config.sh inputboundary"
    exit 1
    fi
    for f in `ls *.config`
    do echo $f
    pin=`awk -F " *= *" '$1=="inputboundary"{print $2}' $f | xargs dirname | xargs dirname`
    pout=`pwd`
    mv $f $f~
    sed '/^ *'$1' *=/s?'$pin'?'$pout'?' $f~ > $f
    done
    echo Now you can: "rm *.config~"
    ;;
"inputraw")
    if [ $# -ne 3 ]; then
    echo "Usage: mpicker_config.sh inputraw /absolute/path/before /absolute/path/after"
    exit 1
    fi
    for f in `ls *.config`
    do echo $f
    pin=$2
    pout=$3
    mv $f $f~
    sed '/^ *'$1' *=/s?'$pin'?'$pout'?' $f~ > $f
    done
    echo Now you can: "rm *.config~"
    ;;
"inputmask")
    if [ $# -ne 3 ]; then
    echo "Usage: mpicker_config.sh inpumask /absolute/path/before /absolute/path/after"
    exit 1
    fi
    for f in `ls *.config`
    do echo $f
    pin=$2
    pout=$3
    mv $f $f~
    sed '/^ *'$1' *=/s?'$pin'?'$pout'?' $f~ > $f
    done
    echo Now you can: "rm *.config~"
    ;;
"back")
    if [ $# -ne 1 ]; then
    echo "Usage: mpicker_config.sh back"
    exit 1
    fi
    for f in `ls *.config~`
    do echo $f
    mv $f `basename $f .config~`.config
    done
    ;;
*)
    echo "Usage: mpicker_config.sh inputboundary  # assume .config file is in current directory"
    echo "OR: mpicker_config.sh inputraw /absolute/path/before /absolute/path/after  # just replace the string by sed"
    echo "OR: mpicker_config.sh inputmask /absolute/path/before /absolute/path/after  # just replace string by sed"
    echo "OR: mpicker_config.sh back  # mv *.config~ to *.config"
    ;;
esac

exit 0
