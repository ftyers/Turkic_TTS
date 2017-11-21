for f in trim_clean_*.flac; do
	p=`echo $f | cut -f1 -d'.'`
	pf=$p".wav"
	sox $f /tmp/$pf
	for i in `../../scripts/audio_segmenter.py /tmp/$pf`; do 
		echo $f" "$i; 
		id=`echo $i | cut -f1 -d','`;
		start=`echo $i | cut -f2 -d','`;
		end=`echo $i | cut -f3 -d','`;
		sox $f ../split/$p.$id.flac trim $start =$end 
	done
done

