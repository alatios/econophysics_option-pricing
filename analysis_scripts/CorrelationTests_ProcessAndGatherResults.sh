#!/bin/bash
#
#	Joins completely verbose output of CorrelationTest.cu in an astropy.io-readable table.
#

INPUTFILE="../libraries/CoreLibraries/RandomGenerator/CorrelationTests_UnprocessedOutput.dat"
OUTPUTFILE_INTRASTREAM="../outputs/CorrelationTests/CorrelationTests_MainIntraStream.dat"

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

## INTRASTREAM CORRELATION

OUTPUTFILE_AUTOCORR="../outputs/CorrelationTests/CorrelationTests_Autocorrelations.dat"

LINECOUNT=$(sed -n '/Uniform autocorrelations i\/i+k/,/Gaussian autocorrelations i\/i+k/p' ${INPUTFILE} | head -n -2 | tail -n +2 | sed '/thread/d' | wc -l)
ITERATIONS=$((${LINECOUNT}/512))

rm -f autocorr_threads.dat
for ((thread=0; thread<512; ++thread)); do
	for ((i=0; i<${ITERATIONS}; ++i)); do
		echo ${thread} >> autocorr_threads.dat;
	done
done

sed -n '/Uniform autocorrelations i\/i+k/,/Gaussian autocorrelations i\/i+k/p' ${INPUTFILE} \
	| head -n -2 \
	| tail -n +2 \
	| sed '/thread/d' \
	| sed -e 's/<xi\*xi+//g' \
	| sed -e 's/>:\t/ /g' \
	> autocorr_uni.dat;

sed -n '/Gaussian autocorrelations i\/i+k/,/INTER-STREAM/p' ${INPUTFILE} \
	| head -n -2 \
	| tail -n +2 \
	| sed '/thread/d' \
	| sed -e '/^.*/s/^.*\(:\t\)//' \
	> autocorr_gauss.dat

paste -d" " autocorr_threads.dat autocorr_uni.dat autocorr_gauss.dat > ${OUTPUTFILE_AUTOCORR}
sed -i '1 i\thread offset unicorr gausscorr' ${OUTPUTFILE_AUTOCORR};
rm autocorr_threads.dat autocorr_uni.dat autocorr_gauss.dat;

## INTERSTREAM CORRELATION

OUTPUTFILE_INTERCORR="../outputs/CorrelationTests/CorrelationTests_InterstreamAutocorrelations.dat"

sed -n '/Uniform super stream/,/Gaussian super stream/p' ${INPUTFILE} \
	| head -n -1 \
	| tail -n +4 \
	| sed -e 's/<xi\*xi+//g' \
	| sed -e 's/>:\t/ /g' \
	> inter_autocorr_uni.dat

sed -n '/Gaussian super stream autocorr/,$p' ${INPUTFILE} \
	| tail -n +2 \
	| sed -e '/^.*/s/^.*\(:\t\)//' \
	> inter_autocorr_gauss.dat

paste -d" " inter_autocorr_uni.dat inter_autocorr_gauss.dat > ${OUTPUTFILE_INTERCORR};
sed -i '1 i\offset unicorr gausscorr' ${OUTPUTFILE_INTERCORR};

rm inter_autocorr_uni.dat inter_autocorr_gauss.dat;

exit 0
