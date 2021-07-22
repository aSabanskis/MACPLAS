#!/bin/bash

source ./helper.sh

# dim, order
initialize

r=results-transient

if [[ -f $r/$probes ]]
then
    echo "$r/$probes" exists, remove to rerun.
    exit 0
fi

clean_results

cp parameters/*.prm .

mkdir -p "$r"

sed -Ei "s|(set Temperature only *= *).*|\1 true|" problem.prm
sed -Ei "s|(set Time step *= *).*|\1 2|" temperature.prm

cp -- *.prm "$r"

./macplas-inductive-heating "$dim"d order "$order" > $r/log
./plot-transient-temperature-time.gnu

mv -- *-"$dim"d-order"$order"* probes*.txt "$r"
