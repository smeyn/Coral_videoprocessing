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


"""filter the detection json object to list only interesting videos"""

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite  --labels models/coco_labels.txt

    parser.add_argument(
        "-i", "--input", required=True, help="File path to detection json file",
    )
    parser.add_argument("-o", "--output", required=True, help="path to result.")
    parser.add_argument(
        "-l",
        "--label",
        type=str,
        nargs="+",
        # action="extend",
        help="label of interest.",
    )

    args = parser.parse_args()
    return args


def matches(rec, labels):
    """returns True if rec has these labels.
    rec: dictionary like:
        "2019-10-17_08-37-38-right_repeater.mp4": {
        "file": "2019-10-17_08-37-38-right_repeater.mp4",
        "car": {
            "label": "car",
            "count": 34,
            "frames": [
            0,
            0]
        "person": {...}
    labels: list of strings
    """
    # print(rec)
    for label in labels:
        if label in rec.keys():
            return True
    return False


def filter(rec, labels):
    """returns record with only detected objects that appear in label list"""
    count = 0
    new_rec = {
        "file": rec["file"],
    }
    for label in labels:
        if label in rec.keys():
            count += 1
            new_rec[label] = rec[label]
    if count:
        return new_rec
    else:
        return None


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.input):
        print(f"Error inout file {args.input} not fond")
    else:
        print(args)
        with open(args.input) as fp:
            records = json.load(fp)
        filtered = {
            key: filter(record, args.label)
            for key, record in records.items()
            if matches(record, args.label)
        }
        with open(args.output, "w") as fp_out:
            json.dump(filtered, fp_out)
        print(f"Filtered {len(records)} source records into {len(filtered)} records")

