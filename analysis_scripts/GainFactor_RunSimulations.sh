#!/bin/bash
## Runs simulations cycling through numbers of GPU blocks.

DIRECTORY="GainFactor_Cabibbo_CPU"
NOFSIMSPERTHREAD=1000
MARRAY=(1 10 50 100 200 300 400)

cd ..;
rm -rf outputs/${DIRECTORY}/output*;
for m in ${MARRAY[@]}; do
	for i in {1..88..1}; do
		echo "${i}, ${m}";
		sed -e "s/_gpu_blocks_/$i/g" \
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
			-e "s/_simulations_/$((NOFSIMSPERTHREAD * 512 * i))/g" \
			-e "s/_cpugpu_/c/g" \
			-e "s/_gauss_bimodal_/g/g" \
			input.dat.template | tee input.dat "inputs/${DIRECTORY}/input__${i}_${m}.dat" > /dev/null;
	#	./main.x > "outputs/${DIRECTORY}/output__${i}_${m}.dat";
	done
done

exit 0
