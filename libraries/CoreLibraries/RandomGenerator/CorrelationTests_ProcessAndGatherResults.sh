#!/bin/bash
#
#	Joins completely verbose output of CorrelationTest.cu in an astropy.io-readable table.
#

INPUTFILE="CorrelationTests_UnprocessedOutput.dat"
OUTPUTFILE_INTRASTREAM="CorrelationTests_MainIntraStream.dat"

sed -n '/Uniform averages (exp. 0.5):/,/Gaussian averages (exp. 0):/p' ${INPUTFILE} \
	| head -n -2 \
	| tail -n +2 \
	| awk '{gsub("<xi>@thread","");print}' \
	| awk '{gsub(":\t"," ");print}' \
	> uniformavgs.dat;

sed -n '/Gaussian averages (exp. 0):/,/Gaussian variances (exp. 1)/p' ${INPUTFILE} \
	| head -n -2 \
	| tail -n +2 \
	| sed -e '/^.*/s/^.*\(:\t\)//' \
	> gaussavgs.dat;

sed -n '/Gaussian variances (exp. 1)/,/Gaussian kurtosises/p' ${INPUTFILE} \
	| head -n -2 \
	| tail -n +2 \
	| sed -e '/^.*/s/^.*\(:\t\)//' \
	> gaussvar.dat;

sed -n '/Gaussian kurtosises/,/Uniform autocorrelations/p' ${INPUTFILE} \
	| head -n -2 \
	| tail -n +2 \
	| sed -e '/^.*/s/^.*\(:\t\)//' \
	> gausskurt.dat;

paste -d" " uniformavgs.dat gaussavgs.dat gaussvar.dat gausskurt.dat > ${OUTPUTFILE_INTRASTREAM};
sed -i '1 i\thread uniavg gaussavg gaussvar gausskurt' ${OUTPUTFILE_INTRASTREAM};
rm uniformavgs.dat gaussavgs.dat gaussvar.dat gausskurt.dat;

OUTPUTFILE_AUTOCORR="CorrelationTests_Autocorrelations.dat"

LINECOUNT=$(sed -n '/Uniform autocorrelations i\/i+k/,/Gaussian autocorrelations i\/i+k/p' ${INPUTFILE} | head -n -2 | tail -n +2 | sed '/thread/d' | wc -l)
ITERATIONS=$((${LINECOUNT}/512))

rm -f autocorr_threads.dat
for ((thread=0; thread<512; ++thread)); do
	for ((i=0; i<${ITERATIONS}; ++i)); do
		echo ${thread} >> autocorr_threads.dat;
	done
done

sed -n '/Uniform autocorrelations i\/i+k/,/Gaussian autocorrelations i\/i+k/p' CorrelationTests_UnprocessedOutput.dat \
	| head -n -2 \
	| tail -n +2 \
	| sed '/thread/d' \
	| sed -e 's/<xi\*xi+//g' \
	| sed -e 's/>:\t/ /g' \
	> autocorr_main.dat;

paste -d" " autocorr_threads.dat autocorr_main.dat > ${OUTPUTFILE_AUTOCORR}
sed -i '1 i\thread offset autocorr' ${OUTPUTFILE_AUTOCORR};
rm autocorr_threads.dat autocorr_main.dat;

exit 0
