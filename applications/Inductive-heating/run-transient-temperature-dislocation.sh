#!/bin/bash

source ./helper.sh

# dim, order, threads
initialize # 2 2 0

r=results-transient
dt=1

if [[ -f $r/$probes ]]
then
    echo "$r/$probes" exists, remove to rerun.
    exit 0
fi

clean_results

mkdir -p "$r"

cp parameters/*.prm .

setup_parameters

sed -Ei "s|(set Temperature only *= *).*|\1 false|" problem.prm
sed -Ei "s|(set Time step *= *).*|\1 $dt|" dislocation.prm
sed -Ei "s|(set Time step *= *).*|\1 $dt|" temperature.prm
sed -Ei "s|(set Max time step *= *).*|\1 $dt|" dislocation.prm
sed -Ei "s|(set Min time step *= *).*|\1 0.01|" dislocation.prm

sed -Ei "s|(set Initial dislocation density *= *).*|\1 1e6|" dislocation.prm
sed -Ei "s|(set Critical stress *= *).*|\1 0.1*exp(10.55+10147/T)|" dislocation.prm
sed -Ei "s|(set Include tau_crit for tau_eff\^l *= *).*|\1 true|" dislocation.prm

sed -Ei "s|(set Max dstrain_c *= *).*|\1 1e-6|" dislocation.prm
sed -Ei "s|(set Max relative dN_m *= *).*|\1 0.2|" dislocation.prm
sed -Ei "s|(set Max relative dtau_eff *= *).*|\1 0.2|" dislocation.prm
sed -Ei "s|(set Max relative time step increase *= *).*|\1 0.2|" dislocation.prm
sed -Ei "s|(set Max v\*dt *= *).*|\1 5e-5|" dislocation.prm
sed -Ei "s|(set Time substep *= *).*|\1 0|" dislocation.prm
sed -Ei "s|(set Max time substeps *= *).*|\1 |" dislocation.prm
sed -Ei "s|(set Refresh stress for substeps *= *).*|\1 true|" dislocation.prm

cp -- *.prm "$r"

./macplas-inductive-heating "$dim"d order "$order" > $r/log
./plot-transient-temperature-time.gnu

mv -- *-"$dim"d-order"$order"* probes*.txt "$r"
