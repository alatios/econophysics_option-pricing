#!/bin/bash
## Runs simulations cycling through values of m (number of intervals)
## with either gaussian or bimodal variables.

DIRECTORY="OptionPriceBimodal"
MARRAY=(1 2 3 4 5 6 7 8 9 10 15 20 25 30 50 100 150 200 250 300 350)

cd ..;
rm -rf outputs/${DIRECTORY}/output*;
for m in "${MARRAY[@]}"; do
	for v in b g; do
		echo "${m}, ${v}";
		sed -e "s/_gpu_blocks_/50/g" \
			-e "s/_underlying_initial_price_/100/g" \
			-e "s/_volatility_/0.25/g" \
			-e "s/_riskfreerate_/0.0001/g" \
			-e "s/_ttm_/1/g" \
			-e "s/_intervals_/$m/g" \
			-e "s/_option_type_/c/g" \
			-e "s/_strike_price_/100/g" \
			-e "s/_B_/1/g" \
			-e "s/_K_/0.3/g" \
			-e "s/_N_/1/g" \
			-e "s/_simulations_/10000000/g" \
			-e "s/_cpugpu_/g/g" \
			-e "s/_gauss_bimodal_/$v/g" \
			input.dat.template | tee input.dat "inputs/${DIRECTORY}/input__${m}_${v}.dat" > /dev/null;
		./main.x > "outputs/${DIRECTORY}/output__${m}_${v}.dat";
	done
done

exit 0
