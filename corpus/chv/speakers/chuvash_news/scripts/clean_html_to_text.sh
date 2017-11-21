for i in `cat track-id-list.txt`; do
	python3 scripts/desoupify.py transcripts/html/$i.html > transcripts/raw/$i.txt
	len=`cat transcripts/raw/$i.txt | wc -lw `
	echo $i" "$len
done
