# Data for Chuvash TTS

## Prerequisites

* From Debian/Ubuntu: sudo apt-get install python3-pip python3-numpy python3-scipy python3-sklearn; sudo pip3 install pydub eyeD3 hmmlearn


## Procedure for extracting training data

1. 

In the ./audio/proc subdirectory, run the ../../scripts/clean_audio.sh script. This will 
read audio from ./audio/orig directory and generate cleaned audio in the ./audio/proc directory. It will
also remove the first 3 seconds and the last 3 seconds from the files.

2. 

Again in the ./audio/proc subdirectory run the ../../scripts/segment_audio.sh script. This 
will take the clean files, run them through the audio_segmenter.py script, and then 
split them using sox. The output files are in ./audio/split

3. 

In the ./ directory, run ./scripts/extract_training.sh with an argument of the output directory. This will
copy the sentence and audio for that sentence into separate files with the same prefix but different 
suffixes in the output directory. It will only include those files which are compatible in terms of how
many sentences and audio segments there are.

## Procedure for adding manual segmentation

Sometimes the automatic segmentation does not align properly with the text. In this case 
you can specify the manual segmentation in `./audio/manual`. The manual segmentation
should be in the same format as the output of the automatic segmenter.

### Example

In the `./audio/` subdirectory:

```
$ python3 ../scripts/audio_segmenter.py proc/trim_clean_17289.flac > manual/17289.seg.txt
$ cat manual/17289.seg.txt
0001,1.9400,11.6400
0002,14.2200,21.6000
0003,24.6400,33.5800
0004,35.3800,39.4200
0005,41.8800,46.7200
0006,49.0400,54.2800
0007,57.6800,64.6200
0008,67.8400,71.0000
```

Now edit the file `manual/17289.seg.txt` and fix the segmentation.

