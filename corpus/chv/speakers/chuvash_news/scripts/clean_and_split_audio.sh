#!/bin/bash
# to be run from the /audio/proc subdir
# assuming that the original audio is in ../orig/*.flac


# cut off seconds 1-3 (beginning) of original audio for silence profile
for i in ../orig/*.flac; do
    sox $i begin_silence_${i##*/} trim 1 3;
done

# create special noise profile file from silence selection
for i in begin_silence_*; do
    sox $i -n noiseprof ${i%.flac}.prof;
done

# apply noise reduction to file from its own noise silence profile
for i in ../orig/*.flac; do
    filename=${i##*/};
    sox $i clean_${filename} noisered begin_silence_${filename/%.flac}.prof 0.21;
done
for i in clean_*; do filename=${i%.flac}; sox $i ../split/${filename}_split_.flac silence -l 1 0.1 0.85% 1 1.0 0.85% : newfile : restart; done
for i in clean_*; do filename=${i%.flac}; sox $i ../split/${filename}_split_.flac silence -l 1 0.1 0.85% 1 1.0 0.85% : newfile : restart; done
