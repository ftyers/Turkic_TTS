#!/bin/bash

txt_file=$1
audio_file=$2
objective=`cat $txt_file | wc -l`

for on_dur in .25 .5 .75 1.0 1.25 1.5 1.75 2.0; do
    for on_level .1 .5 1.0 1.5 2.0 2.5 3.0; do
        for off_dur in .25 .5 .75 1.0 1.25 1.5 1.75 2.0; do
            for off_level in .1 .5 1.0 1.5 2.0 2.5 3.0; do
                sox $audio_file file_new_.flac silence -l 1 $on_dur $on_level 1 $off_dur $off_level : newfile : restart
                nfiles=`ls file_new_* | wc -l`
                if [[ $nfiles -eq $objective ]]; then
                    echo "$txt_file $objective $nfiles $on_dur $on_level $off_dur $off_level";
                    break;
                fi
                rm file_new_*;
            done
        done
    done
done
