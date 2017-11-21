for i in `cat track-id-list.txt`; do
	python3 scripts/desoupify.py transcripts/html/$i.html > transcripts/raw/$i.txt
	cat transcripts/raw/$i.txt | python3 scripts/sentence_segmenter.py > transcripts/txt/$i.txt
	len=`cat transcripts/raw/$i.txt | wc -lw `
	len2=`cat transcripts/txt/$i.txt | wc -l`
	echo $i" "$len" "$len2
done
