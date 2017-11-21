# This script calculates the duration of each track and plots a spectrogram
# which can be used for debugging

for i in `cat track-id-list.txt`; do 
	f="audio/orig/"$i".flac"
	sec=0 # How many seconds long the file is
	dur="_" # Duration
	if [[ -e $f ]]; then
		# Get how long the file is
		sec=`metaflac --show-total-samples --show-sample-rate $f | tr '\n' ' ' | awk '{print $1/$2}'`
		# Convert to hh:mm:ss
		dur=`date -u -d @"$sec" +"%T"`
		# Get the first 20 seconds of audio
		sox $f /tmp/$i.flac trim 0 20
		# Plot a spectrogram
		sox /tmp/$i.flac -r 8k -n rate spectrogram -t $f -x 700 -y 300 -l -m -o audio/spectrogram/$i.png
	fi
	echo -e $i"\t"$dur
done

# scp -i "/home/fran/.ssh/jm-aws-tts-data-key.pem" orig/*.flac ubuntu@ec2-35-164-219-223.us-west-2.compute.amazonaws.com:~/Turkic_TTS/corpus/chv/speakers/chuvash_news/audio/orig/
