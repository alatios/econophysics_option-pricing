#!/bin/bash
## Runs simulations cycling through numbers of GPU blocks.

DIRECTORY="InfiniteIntervals"
KARRAY=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
BARRAY=(0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.5 3.0 3.5 4.0)

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
for B in ${BARRAY[@]}; do
	for K in ${KARRAY[@]}; do
		echo "${B}, ${K}";
		sed -e "s/_gpu_blocks_/50/g" \
			-e "s/_underlying_initial_price_/100/g" \
			-e "s/_volatility_/0.25/g" \
			-e "s/_riskfreerate_/0.0001/g" \
			-e "s/_ttm_/1/g" \
			-e "s/_intervals_/10000/g" \
			-e "s/_option_type_/e/g" \
			-e "s/_strike_price_/100/g" \
			-e "s/_B_/$B/g" \
			-e "s/_K_/$K/g" \
			-e "s/_N_/1/g" \
			-e "s/_simulations_/10000000/g" \
			-e "s/_cpugpu_/g/g" \
			-e "s/_gauss_bimodal_/g/g" \
			input.dat.template | tee input.dat "inputs/${DIRECTORY}/input__${B}_${K}.dat" > /dev/null;
		./main.x > "outputs/${DIRECTORY}/output__${B}_${K}.dat";
	done
done

exit 0
