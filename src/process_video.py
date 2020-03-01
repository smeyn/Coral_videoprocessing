# Copyright 2020 Stephanmeyn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Video_Processor processes a video to identify objects usign tensorflow Coral Board.
"""
import argparse
import time
from glob import glob
from PIL import Image
from PIL import ImageDraw
import numpy as np
import os
import time
import detect
import tflite_runtime.interpreter as tflite
import platform
import cv2
import logging
import datetime
import json


EDGETPU_SHARED_LIB = {
    "Linux": "libedgetpu.so.1",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}[platform.system()]


class VideoStream:
    """Processes a video, allowing extraction of singular frames."""

    def __init__(self, video_path, skip=1, markup_path=None, labels={}):
        """
        Args: 
            video_path: string, path to video file
            skip: sampling rate. Default is 1 = sample every frame
            markup: bool if True then  write a processed video 
        """
        logging.debug(
            f"VideoStream({video_path}, skip={skip}, markup_path={markup_path})"
        )
        self.input_path = video_path
        self.vidcap = cv2.VideoCapture(video_path)
        frame_width = round(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = round(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = self.vidcap.get(cv2.CAP_PROP_FOURCC)
        v = int(four_cc)
        decoded_4cc = "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])
        logging.debug(f' 4cc: "{decoded_4cc}""')
        fps = round(self.vidcap.get(cv2.CAP_PROP_FPS))
        logging.debug(
            f"Reading video, fps:{fps}, w:{frame_width}, h:{frame_height}, 4cc: {v}"
        )
        self.count = 0
        self.labels = labels
        self.current_image = None
        if markup_path is not None:
            dir_name, file_name = os.path.split(video_path)
            base_name, ext = os.path.splitext(file_name)
            target_path = os.path.join(markup_path, f"{base_name}_processed{ext}")
            if not os.path.exists(markup_path):
                os.mkdir(markup_path)
            elif not os.path.isdir(markup_path):
                logging.error(
                    f"markup_path is not a directory. Cannot write to {markup_path}"
                )
                return
            logging.debug(f"writing marked up video to {target_path}")
            self.vid_writer = cv2.VideoWriter(
                target_path,
                # v,
                cv2.VideoWriter_fourcc("a", "v", "c", "1"),
                fps,
                (frame_width, frame_height),
            )
        else:
            self.vid_writer = None
        if skip == 0:
            self.skip = 1
        else:
            self.skip = skip
        self.detection_frames = []

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        while True:
            if self.vid_writer and self.current_image is not None:
                marked_up_frame = self._draw_boxes()
                if marked_up_frame is not None:
                    self.vid_writer.write(marked_up_frame)

            success, img = self.vidcap.read()

            if not success:
                self.vidcap.release()
                self.vidcap = None
                if self.vid_writer:
                    self.vid_writer.release()
                    self.vid_writer = None
                raise StopIteration()
            self.count += 1
            if img is None:
                logging.info(f"empty frame :[{self.count}] {self.video_path}")
                continue
            self.current_image = img

            if self.count % self.skip == 0:  # return this image
                try:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    return im_pil, self.count
                except Exception as ex:
                    logging.error(f"error:{self.video_path}. {ex}")
                    pass

    def set_detected_objects(self, detection_frames):
        self.detection_frames = detection_frames

    def _draw_boxes(self):
        """Draws the boxes around the image."""
        if not self.vid_writer:
            return None
        if self.current_image is None:
            return None

        img = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        image = im_pil.convert("RGB")
        draw = ImageDraw.Draw(image)
        # logging.debug(f"drawing {len(self.detection_frames)} boxes")
        for obj in self.detection_frames:
            bbox = obj.bbox
            draw.rectangle(
                [(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)], outline="red"
            )
            draw.text(
                (bbox.xmin + 10, bbox.ymin + 10),
                "%s\n%.2f" % (self.labels.get(obj.id, obj.id), obj.score),
                fill="red",
            )
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


class Video_Processor:
    def __init__(self, model_file, labels):
        """sets up a video detection processor.
        Args:
            model_file: path to model_file
            labels: dictionary that maps object ids to labels
        """
        self.interpreter = self.make_interpreter(model_file)
        self.interpreter.allocate_tensors()
        self.labels = labels
        self.interpreter.invoke()

    def process_video(self, video_path, skip=32, threshold=0.5, markup_path=None):
        """processes a complete video.
        Args:
            video_path: path to input file
            skip: nr of frames to skip. default is 32
            threshold: detection threshold, default =0.5
            markup_path: path to write the marked up video to. If None, no markup is done. 
        Returns:
            dict of detected objects summary
        """
        objects_detected = {}
        logging.debug(f"start to process {os.path.basename(video_path)}")
        video_stream = VideoStream(
            video_path, skip, markup_path=markup_path, labels=self.labels
        )
        for frame, frame_id in video_stream:
            objs = self._detect_image(threshold, frame)
            self._aggregate_detections(objs, self.labels, frame_id, objects_detected)
            video_stream.set_detected_objects(objs)

        for label, count in objects_detected.items():
            logging.debug(f"   {label}: {count}")
        return objects_detected

    def _detect_image(self, threshold, image):
        scale = detect.set_input(
            self.interpreter,
            image.size,
            lambda size: image.resize(size, Image.ANTIALIAS),
        )
        self.interpreter.invoke()
        objs = detect.get_output(self.interpreter, threshold, scale)
        return objs

    def _aggregate_detections(self, objs, labels, frame_id, container):
        """aggregate a single frame detection objects into a result container.
        Args:
            objs: list of detection objects returned by detect
            labels: dict mapping ids to labels
            frame_id: frame nr this detection was made in
            container: dict that contains aggregated results
        Returns:
            updated container

        Note: detection obj structure
            - id
            - score - score of detection
            - bbox - bounding box
        """
        for obj in objs:
            label = labels.get(obj.id, obj.id)
            result_obj = container.get(
                label, {"label": label, "count": 0, "frames": []}
            )
            result_obj["count"] = result_obj["count"] + 1
            result_obj["frames"].append(frame_id)
            container[label] = result_obj
        return container

    def make_interpreter(self, model_file):
        """Instantiates the model interpreter."""
        model_file, *device = model_file.split("@")
        interpreter = tflite.Interpreter(
            model_path=model_file,
            experimental_delegates=[
                tflite.load_delegate(
                    EDGETPU_SHARED_LIB, {"device": device[0]} if device else {}
                )
            ],
        )
        return interpreter


def load_labels(path, encoding="utf-8"):
    """Loads labels from file (with or without index numbers).

  Args:
    path: path to label file.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
    with open(path, "r", encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
            return {}

        if lines[0].split(" ", maxsplit=1)[0].isdigit():
            pairs = [line.split(" ", maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite  --labels models/coco_labels.txt
    parser.add_argument(
        "--model",
        help="File path of .tflite file.",
        default="models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite",
    )
    parser.add_argument(
        "-i", "--input", required=True, help="File path of video to process."
    )
    parser.add_argument(
        "-o", "--output", required=True, help="File path to write the results."
    )
    parser.add_argument(
        "-s", "--skip", default=16, type=int, help="skip every n frames (default=16)"
    )

    parser.add_argument(
        "--markup_path",
        type=str,
        help="set this to mark up video with detection rectangles. results are written to this directory",
        nargs="?",
    )

    parser.add_argument(
        "-l",
        "--labels",
        default="models/coco_labels.txt ",
        help="File path of labels file.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.4,
        help="Score threshold for detected objects.",
    )
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    logging.basicConfig(level=logging.DEBUG, filename="process_video.log")

    result = []
    image_paths = glob(args.input)
    if len(image_paths) == 0:
        print("Nothing to process")
        return -1

    labels = load_labels(args.labels) if args.labels else {}
    # interpreter = make_interpreter(args.model)
    # interpreter.allocate_tensors()

    print("----INFERENCE TIME----")
    print(
        "Note: The first inference is slow because it includes",
        "loading the model into Edge TPU memory.",
    )
    batch_start_ts = datetime.datetime.now()

    # interpreter.invoke()
    COUNT = "Frames Processed"
    fileCount = 0
    allStart = datetime.datetime.now()
    results = {}
    processor = Video_Processor(args.model, labels)
    logging.debug(f"Start: {datetime.datetime.now()}")
    try:
        for image_path in image_paths:
            image_start_ts = datetime.datetime.now()
            logging.debug(f"{os.path.basename(image_path)}")
            objects_list = processor.process_video(
                image_path, args.skip, markup_path=args.markup_path
            )
            # add the file name to the entry
            objects_list["file"] = os.path.basename(image_path)
            results[os.path.basename(image_path)] = objects_list
            fileCount += 1
            delta = datetime.datetime.now() - image_start_ts
            totalElapsed = datetime.datetime.now() - batch_start_ts
            logging.info(
                f"File Nr# {fileCount}.  {delta.total_seconds()} seconds. Total time elapsed = {totalElapsed}"
            )
    finally:
        with open(args.output, "w") as fp:
            json.dump(results, fp)

    logging.debug(f"End: {datetime.datetime.now()}")


if __name__ == "__main__":
    main()
