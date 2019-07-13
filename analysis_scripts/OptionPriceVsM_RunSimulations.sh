#!/bin/bash
## Runs simulations cycling through values of N (number of simulations)
## and m (number of intervals).

NARRAY=(1000 10000 100000 1000000 10000000 100000000)
MARRAY=(1 2 3 4 5 10 50 100 150 200 250 300 400)
DIRECTORY="OptionPriceVsM"

echo "This will overwrite all existing output. Are you sure you want to run this? y/[n]"
read response

case "$response" in
	[yY][eE][sS]|[yY])
		:
		;;
	*)
		echo "Aborting..."
		exit 3
		;;
esac

cd ..;
rm -rf outputs/${DIRECTORY}/output*;
rm -rf inputs/${DIRECTORY}/input*;

for N in "${NARRAY[@]}"; do
	for m in "${MARRAY[@]}"; do
		echo "${m}, ${N}";
		sed -e "s/_gpu_blocks_/50/g" \
			-e "s/_underlying_initial_price_/100/g" \
			-e "s/_volatility_/0.25/g" \
			-e "s/_riskfreerate_/0.0001/g" \
			-e "s/_ttm_/1/g" \
			-e "s/_intervals_/$m/g" \
			-e "s/_option_type_/e/g" \
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
