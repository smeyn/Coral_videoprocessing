# Video processing using the Coral Edge USB board

This is a demo I did to process the Sentry Videos for a Tesla.

## Background

The Tesla has a Sentry mode whih can be activated when the car is parked. If someone gets too close to the car 4 cameras will start up and record about 60 seconds of videro. If Sentry determines here is still some one close by it will record futher 60 sconds clips.  
So it is not uncommmon to accumulate lots of video clips which take too long to manually review.
So I wrote a sample script that will use the object detection model from the CORAL boards example directory to detect objects in the video.
There is an opttion to have the videos marked up with the rectangles around the detected objects

## How it works

The key class is Video_Processor. It is initialised with the model file for object detectioon. Then it can process videos.
Options to process the video are:

- path to video file
- how many frames to skip before processing a video (default 32)
- detection threshold (default=0.5)
- path to directory to keep the marked up videos (can be None to avoid marking up)

Once it processed the video it returns a dictionary of all the detection events keyed by the detected object category. The value to the key is a list of the frames

## Prerequisites:

### 1. Edge TPU setup

https://coral.ai/docs/accelerator/get-started/

### 2 : get detect.py file

from https://github.com/google-coral/tflite/blob/master/python/examples/detection/detect.py

and copy it to the same directory where your code runs

### 3: get model files

follow the instructions (or run the script) to get the model file
https://github.com/google-coral/tflite/blob/master/python/examples/detection/install_requirements.sh

### 3: get OpenH264 Video Codec

This is optional and only needed if you want to mark up the video. If you don't install it,
the markup will stll work but the resulting video will be large.
get it from: https://github.com/cisco/openh264/releases

Copy it into the same directory where your code runs

## Running

The file process_video.py can be invoked from the command line.
Running it without --help provides guidance:

```
process_video.p" --help
usage: process_video.py [-h] [--model MODEL] -i INPUT -o OUTPUT [-s SKIP]
                        [--markup_path [MARKUP_PATH]] [-l LABELS]
                        [-t THRESHOLD]

optional arguments:
   -h, --help            show this help message and exit
  --model MODEL         File path of .tflite file. (default: models/mobilenet_
                        ssd_v2_coco_quant_postprocess_edgetpu.tflite)
  -i INPUT, --input INPUT
                        File path of video to process. (default: None)
  -o OUTPUT, --output OUTPUT
                        File path to write the results. (default: None)
  -s SKIP, --skip SKIP  skip every n frames (default=16) (default: 16)
  --markup_path [MARKUP_PATH]
                        set this to mark up video with detection rectangles.
                        results are written to this directory (default: None)
  -l LABELS, --labels LABELS
                        File path of labels file. (default:
                        models/coco_labels.txt )
  -t THRESHOLD, --threshold THRESHOLD
                        Score threshold for detected objects. (default: 0.4)
```

## Performance

I batch processed a total of 2900 60 second clips:

- that is a total of over 48 hours of video
- It ran on an INTEL NUC I7 in a single thread
- skipping every 30 frames (i.e. analysing at 1 second intervals)
- processed 2900\*60 = 174,000 frames
- it took 10 hours = so it averaged 4.8 seconds per image 174
