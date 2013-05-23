#!/usr/bin/env python
"""
Capture and save first N frames of a video.
"""
import cv
import sys

files = sys.argv[1]
N = sys.argv[2]

for f in files:
    capture = cv.CaptureFromFile(f)
    print capture

    print cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH)
    print cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT)

    for i in range(N):
        frame = cv.QueryFrame(capture)
        if frame:
            print frame
            cv.SaveImage('frames/frame'+str(i)+'.png', frame)
