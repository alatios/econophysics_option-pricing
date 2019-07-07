#!/bin/bash
## Runs simulations cycling through values of N (number of simulations)
## and m (number of intervals).

NARRAY=(5000000 10000000 50000000 100000000 200000000)
MARRAY=(1 2 3 4 5 10 50 100 150 200 250 365)

cd ..;
rm -rf outputs/output_PriceVsM/output*;
for N in "${NARRAY[@]}"; do
	for m in "${MARRAY[@]}"; do
		echo "${m}, ${N}";
		sed -e "s/_gpu_blocks_/10/g" \
			-e "s/_underlying_initial_price_/100/g" \
			-e "s/_volatility_/0.25/g" \
			-e "s/_riskfreerate_/0.01/g" \
			-e "s/_ttm_/1/g" \
			-e "s/_intervals_/$m/g" \
			-e "s/_option_type_/c/g" \
			-e "s/_strike_price_/100/g" \
			-e "s/_B_/1/g" \
			-e "s/_K_/0.3/g" \
			-e "s/_N_/1/g" \
			-e "s/_simulations_/$N/g" \
			-e "s/_cpugpu_/g/g" \
			-e "s/_gauss_bimodal_/g/g" \
			input.dat.template | tee input.dat "inputs/OptionPriceVsM/input__${m}_${N}.dat" > /dev/null;
		./main.x > "outputs/OptionPriceVsM/output__${m}_${N}.dat";
	done
done

exit 0
