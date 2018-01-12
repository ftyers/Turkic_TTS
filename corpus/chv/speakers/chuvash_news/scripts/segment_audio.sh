# For each of the trimmed and cleaned files
for f in trim_clean_*.flac; do
	p=`echo $f | cut -f1 -d'.'`
	ix=`echo $f | cut -f1 -d'.' | sed 's/trim_clean_//g'`

	# If there is a manual segmentation then use it
	if [[ -e ../manual/$ix.seg.txt ]]; then
		echo -e "[m] $f"
		cat ../manual/$ix.seg.txt > /tmp/$p.seg			
	else
	# Otherwise generate a new segmentation automatically
		echo -e "[a] $f"
		python3 ../../scripts/audio_segmenter.py $f > /tmp/$p.seg
	fi

	# Run through each line in the segmentation and use
	# sox to split the file according to the start/end points
	for i in `cat /tmp/$p.seg`; do 
		echo $f" "$i; 
		id=`echo $i | cut -f1 -d','`;
		start=`echo $i | cut -f2 -d','`;
		end=`echo $i | cut -f3 -d','`;
		sox $f ../split/$p.$id.flac trim $start =$end 
	done
done

