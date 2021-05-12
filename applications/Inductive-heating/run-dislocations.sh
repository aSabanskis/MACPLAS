#!/bin/bash

set -e # exit script on error

cp parameters/*.prm .

sed -Ei "s|(set Load saved results *= *).*|\1 true|" problem.prm

# Inductor currents, A and dislocation densities, m^-2
arr_I=(20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120)
arr_N0=(1e4 1e5 1e6 1e7 1e8 1e9)

for i in "${!arr_I[@]}"; do
	for j in "${!arr_N0[@]}"; do
		I=${arr_I[$i]}
		N=${arr_N0[$j]}
		s="I$I"
		r="I$I-disl-$N"

		if [[ ! -d "$s" ]]; then
			echo "$s" does not exist, skipping
			continue
		fi

		if [[ -d "$r" ]]; then
			echo "$r" exists, skipping
			continue
		fi

		echo "$r" "$s"
		cp "$s"/*3d* .

		# start N_m defined in .prm
		rm -- *density* *t[1-9]*
		sed -Ei "s|(set Initial dislocation density *= *).*|\1 $N|" dislocation.prm
		./macplas-inductive-heating > log
		./plot-probes-minmax.gnu

		mkdir "$r"
		cp -- *t[1-9]* *probes* *.prm log "$r"
	done
done
