#!/bin/bash
## Runs simulations cycling through number of intervals
## to identify paths producing negative prices in Euler formula.

DIRECTORY="NegativePrices"
MARRAY=(1 2 3 4 5 6 7 8 9 10 20 30 40)
VARRAY=(0 0.25 0.5 0.75 1)

cd ..;
rm -rf outputs/${DIRECTORY}/output*;
rm -rf inputs/${DIRECTORY}/input*;

for v in ${VARRAY[@]}; do
	for m in ${MARRAY[@]}; do
		echo "${v}, ${m}";
		sed -e "s/_gpu_blocks_/50/g" \
			-e "s/_underlying_initial_price_/100/g" \
			-e "s/_volatility_/$v/g" \
			-e "s/_riskfreerate_/0.0001/g" \
			-e "s/_ttm_/1/g" \
			-e "s/_intervals_/$m/g" \
			-e "s/_option_type_/e/g" \
			-e "s/_strike_price_/100/g" \
			-e "s/_B_/1/g" \
			-e "s/_K_/0.3/g" \
			-e "s/_N_/1/g" \
			-e "s/_simulations_/20000000/g" \
			-e "s/_cpugpu_/g/g" \
			-e "s/_gauss_bimodal_/g/g" \
			input.dat.template | tee input.dat "inputs/${DIRECTORY}/input__${v}_${m}.dat" > /dev/null;
		./main.x > "outputs/${DIRECTORY}/output__${v}_${m}.dat";
	done
done

exit 0
