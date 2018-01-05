#!/usr/bin/env bash
# Produce training data and output to a directory
# Only include those files where the number of audio 
# segments matches the number of sentences in the text

output_dir=$1
mkdir -p $output_dir/txt $output_dir/wav
err_file=$output_dir"_err.log"

if [[ $# -lt 1 ]]; then
	echo "bash scripts/extract_training.sh <target directory>"
	exit;
fi

if [[ -d $1 ]]; then
	total=0
	proc=0
	sec=0
	dur=0
	for id in `cat track-id-list.txt`; do
		total=`expr $total + 1`;
		if [[ ! -f transcripts/txt/$id.txt ]]; then
			continue
		fi
		nf=`ls audio/split/trim_clean_$id.*.flac 2>/dev/null| wc -l`;
		if [[ $nf -eq 0 ]]; then
			continue;
		fi
		txt_sent_no=`cat transcripts/txt/$id.txt | wc -l`;
		audio_files_no=`ls -1 audio/split/trim_clean_$id.*.flac | wc -l`
		echo -ne $id"\t"$txt_sent_no"\t"$audio_files_no"\t";
	
		if [[ $txt_sent_no -eq $audio_files_no ]]; then
			for sid in `cat transcripts/txt/$id.txt | cut -f1`; do 
				cat transcripts/txt/$id.txt | grep -P "^$sid\t" | cut -f2 > $output_dir/txt/$id.$sid.txt
				sox audio/split/trim_clean_$id.$sid.flac $output_dir/wav/$id.$sid.wav
				len=`sox $output_dir/wav/$id.$sid.wav -n stat 2>&1 | grep 'Leng' | cut -f2 -d':' | tr -d ' '`;
				sec=`echo "$sec + $len" | bc -l`;
			done	
			sdur=`date -u -d @"$sec" +"%T"`
			proc=`expr $proc + 1`;
			echo -e $sec"\t"$sdur"\t"$proc"/"$total;
		else
			echo $id >> $err_file;
			echo ""
		fi
	done
	dur=`date -u -d @"$sec" +"%T"`
	echo $dur
	echo $sec
	echo $proc"/"$total
else
	echo "No such directory $1";
fi;



for ext in "txt" "wav"; do 

    cd $output_dir/$ext

    for i in *$ext; do
	
	filename=$(echo ${i%.$ext} | sed 's/\./-/g'); mv $i $filename.$ext;

    done

    cd ../..
    	    
done


