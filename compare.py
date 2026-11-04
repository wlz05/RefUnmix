# -*- coding: utf-8 -*-
# RefUnmix: Separating Prompt Response from Shared Transport in One-Step Image Editing

import argparse
import json
from engine.comparison import BACKBONES, METHODS, Registry

def main():
    parser = argparse.ArgumentParser(description="Prepare the method-by-backbone comparison without running experiments.")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--backbones", nargs="+", choices=BACKBONES, default=list(BACKBONES))
    parser.add_argument("--adapters", nargs="*", default=[])
    parser.add_argument("--turbo", default="models")
    parser.add_argument("--brush", default="models/brush")
    parser.add_argument("--insta", default="models/insta")
    args = parser.parse_args()
    registry = Registry()
    for path in args.adapters:
        registry.load(path)
    plan = registry.plan(args.methods, args.backbones, {"sdturbo": args.turbo, "swiftbrush": args.brush, "instaflow": args.insta})
    print(json.dumps(plan, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
