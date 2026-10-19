# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import argparse
import json
from engine.corpus import PROFILES, convert, profile

def parser():
    result = argparse.ArgumentParser(description="Prepare local editing benchmarks without downloads.")
    result.add_argument("--kind", type=profile, choices=tuple(PROFILES), required=True)
    result.add_argument("--input", default=None)
    result.add_argument("--dataset", default="./data")
    result.add_argument("--output", default="./prepared")
    result.add_argument("--captions", default=None)
    result.add_argument("--partition", type=str.casefold, choices=("train", "val", "test"), default=None)
    result.add_argument("--template", action="store_true")
    return result

def main():
    args = parser().parse_args()
    if args.template:
        print(json.dumps({"kind": args.kind, "caption_sidecar": {"record-key": {"source": "A blue shirt on a person.", "target": "A striped shirt on a person."}}, "requirements": "Use actual captions, relative image paths, and official partition tags."}, indent=2))
        return
    if not args.input:
        raise ValueError("Specify a local annotation file or saved dataset.")
    print(json.dumps(convert(args.kind, args.input, args.dataset, args.output, args.captions, args.partition), allow_nan=False))

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
