for i in clean_*.flac; do dur=`soxi -D $i`; sox $i /tmp/$i trim 3 `echo "$dur-3.0" | bc -l`; done
for i in clean_*.flac; do sox /tmp/$i trim_${i##*/} trim 0 -3; done
