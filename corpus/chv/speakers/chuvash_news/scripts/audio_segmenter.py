#!/usr/bin/python

import sys
import audioBasicIO as aIO
import audioSegmentation as aS

[Fs, x] = aIO.readAudioFile(sys.argv[1])
segments = aS.silenceRemoval(x, Fs, 0.020, 0.020, smoothWindow = 1.0, Weight = 0.3, plot = False)

sid = 1
for i in segments:
    print('%s,%.4f,%.4f' % (str(sid).zfill(4),i[0],i[1]))
    sid += 1
