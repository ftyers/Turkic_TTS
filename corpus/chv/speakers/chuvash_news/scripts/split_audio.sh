
#for i in clean_*; do 
for i in clean_17410*; do 
	echo $i
	filename=${i%.flac}; 
	#                                                remove_silence  duration    sensitivity
	sox $i ../split/${filename}_split_.flac silence -l 1 0.3 0.5% 1 0.3 0.5% : newfile : restart; 
done

