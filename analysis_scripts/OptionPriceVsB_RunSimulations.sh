#!/bin/bash
## Runs simulations cycling through values of N (number of simulations)
## and B (corridor barrier).

NARRAY=(1000 5000 10000 50000 100000 500000 1000000 5000000 10000000 50000000 100000000 200000000)
BARRAY=(0 0.25 0.5 0.75 1 1.5 2 3 4 )
DIRECTORY="OptionPriceVsB"

cd ..;
rm -rf outputs/${DIRECTORY}/output*;
for N in "${NARRAY[@]}"; do
	for B in "${BARRAY[@]}"; do
		echo "${B}, ${N}";
		sed -e "s/_gpu_blocks_/50/g" \
			-e "s/_underlying_initial_price_/100/g" \
			-e "s/_volatility_/0.25/g" \
			-e "s/_riskfreerate_/0.01/g" \
			-e "s/_ttm_/1/g" \
			-e "s/_intervals_/365/g" \
			-e "s/_option_type_/c/g" \
			-e "s/_strike_price_/100/g" \
			-e "s/_B_/$B/g" \
			-e "s/_K_/0.3/g" \
			-e "s/_N_/1/g" \
			-e "s/_simulations_/$N/g" \
			-e "s/_cpugpu_/g/g" \
			-e "s/_gauss_bimodal_/g/g" \
			input.dat.template | tee input.dat "inputs/${DIRECTORY}/input__${B}_${N}.dat" > /dev/null;
		./main.x > "outputs/${DIRECTORY}/output__${B}_${N}.dat";
	done
done

exit 0
