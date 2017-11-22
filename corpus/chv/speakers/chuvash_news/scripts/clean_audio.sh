#!/bin/bash
# to be run from the /audio/proc subdir
# assuming that the original audio is in ../orig/*.flac

echo "Cutting off silence profile..."
# cut off seconds 1-3 (beginning) of original audio for silence profile
for i in ../orig/*.flac; do
    sox $i begin_silence_${i##*/} trim 1 3;
done

echo "Create noise profile..."
# create special noise profile file from silence selection
for i in begin_silence_*.flac; do
    sox $i -n noiseprof ${i%.flac}.prof;
done

echo "Apply noise reduction..."
# apply noise reduction to file from its own noise silence profile
for i in ../orig/*.flac; do
    filename=${i##*/};
    # 0.21 is how much noise gets bleached
    sox $i clean_${filename} noisered begin_silence_${filename/%.flac}.prof 0.21;
done

echo "Trim the files..."
# take the cleaned files and trim off the first 3 seconds
for i in clean_*.flac; do 
	dur=`soxi -D $i`; 
	sox $i /tmp/$i trim 3 `echo "$dur-3.0" | bc -l`; 
done

# take the cleaned files and trim off the last 3 seconds
for i in clean_*.flac; do 
	sox /tmp/$i trim_${i##*/} trim 0 -3; 
done

