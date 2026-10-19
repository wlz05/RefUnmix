# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
from types import SimpleNamespace
from data import clean, destination, locate
from engine.attainment import choose
from engine.serial import read, write

def export(summary, output):
    # Export frozen same-pool choices without new inference.
    source = locate(Path("."), clean(summary, "summary"))
    payload = read(source)
    if payload.get("schema") != 1 or payload.get("split") not in {"val", "test"} or not isinstance(payload.get("rows"), list) or not payload["rows"]:
        raise ValueError("Replay requires a grouped validation or test summary.")
    prepared, identities, filenames = [], set(), set()
    for original in payload["rows"]:
        if not isinstance(original.get("key"), str) or not original["key"] or type(original.get("seed")) is not int or not 0 <= original["seed"] < 2 ** 64:
            raise ValueError("Replay rows require explicit source and seed identities.")
        identity = (original["key"], original["seed"])
        if identity in identities:
            raise ValueError("Replay identities must be unique.")
        identities.add(identity)
        filename = Path(clean(original["image"], "image"))
        if filename.name != str(filename) or filename.suffix.casefold() != ".png" or filename.as_posix().casefold() in filenames:
            raise ValueError("Replay outputs need unique flat PNG names.")
        filenames.add(filename.as_posix().casefold())
        report = original["report"]
        if not report.get("candidatearchive", {}).get("enabled") or not report.get("commonpool"):
            raise ValueError("Every replay row needs archived common-pool choices.")
        pool = [report["raw"], *report["candidates"]]
        candidates = {row["key"]: row for row in pool}
        if len(candidates) != len(pool):
            raise ValueError("Replay candidate identities must be unique.")
        for policy, folder in (("attain", "attained"), ("protect", "protected")):
            choice = report["commonpool"][policy]
            frozen, detail = choose([SimpleNamespace(**row) for row in pool], report["config"]["gainloss"], policy)
            if choice != {"selected": frozen.key, **detail}:
                raise ValueError("Replay choices disagree with frozen candidate scores.")
            selected = candidates.get(choice["selected"])
            if selected is None or not selected["feasible"] or not selected["complete"]:
                raise ValueError("Replay choices must be completed feasible candidates.")
            artifact = "raw" if selected["key"] == "raw" else "candidate" + selected["key"]
            path = locate(Path("."), clean(report["artifacts"][artifact], "artifact"))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != report.get("artifactsha256", {}).get(artifact):
                raise ValueError("Replay artifact hashes do not match frozen images.")
            row = deepcopy(original)
            row["metrics"] = {}
            row["image"] = filename.as_posix()
            row["resultsha256"] = digest
            row["report"].update({"selected": selected["key"], "point": selected["point"], "fallback": selected["key"] == "raw", "scores": selected["scores"], "damage": selected["damage"], "objective": selected["objective"], "penalty": selected["penalty"], "constraintmet": True, "attainment": {**report["attainment"], **choice}, "scope": "Frozen common-pool choice without new generator calls.", "replay": {"source": source.as_posix(), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "policy": policy, "generatorqueries": 0, "cost": "Parent shared-pool generation cost retained; no standalone speed claim."}})
            row["report"]["damagechange"] = selected["damage"] - report["rawdamage"]
            row["report"]["gain"] = sum(selected["scores"].values()) - sum(report["rawscores"].values())
            row["report"]["config"]["selection"] = policy
            prepared.append((folder, path, row))
    root = destination(output)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    result = {}
    for folder in ("attained", "protected"):
        target = root / folder
        target.mkdir()
        rows = []
        for name, path, row in prepared:
            if name == folder:
                shutil.copyfile(path, target / row["image"])
                rows.append(row)
        exported = {key: deepcopy(value) for key, value in payload.items() if key not in {"rows", "summary", "categories", "cost"}}
        exported.update({"rows": rows, "replay": {"source": source.as_posix(), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "scope": "Frozen same-pool selector ablation; scores and labels were not used to regenerate candidates.", "generationcost": payload.get("cost"), "metricnote": "Parent final metrics were discarded; score exported images independently."}})
        write(target / "summary.json", exported)
        result[folder] = (target / "summary.json").as_posix()
    return result

def main():
    parser = argparse.ArgumentParser(description="Export archived same-pool attainment and protection choices.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(export(args.summary, args.output))

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
