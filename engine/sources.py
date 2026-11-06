# -*- coding: utf-8 -*-
# RefUnmix: Separating Prompt Response from Shared Transport in One-Step Image Editing

from pathlib import Path
import subprocess
from engine.reproduce import digest

RELEASES = {
    "pie": ("cure-lab/PnPInversion", "07f97f448150e2ca220bebd54c8f687c5c50c67a"),
    "chord": ("ChordEdit/ChordEdit", "a33d8cc26108851c6545b998c0e939006f89c9bd"),
    "where": ("Xi-Mu-Yu/WhereEdit", "189188cda031fef6ad18bb8027373db08f1402b7"),
    "rrls": ("yhz5613813/RRLS", "c2f88ee62bf4a2a3c5ae94c8eb3449c73d001331"),
    "instant": ("Supercomputing-System-AI-Lab/InstantEdit", "a09fd6d3939e41b7a61209622edace85346fcda9"),
    "infedit": ("sled-group/InfEdit", "eaac91e139cbb6736934b4be7954fb2ae04bbb0b"),
    "turbo": ("GiilDe/turbo-edit", "ebf545e15a858e1917ec2d52f03f177b44375dc9"),
    "swift": ("Qualcomm-AI-research/SwiftEdit", "230bff1aaff5d952210c717d55537c5b29d596ad"),
    "flow": ("fallenshock/FlowEdit", "2620a9364f0f9c21368c36266e6200ff250273ec"),
}
ROLES = {"masactrl": "pie", "pnp": "pie", "flowedit": "flow", "infedit": "infedit", "instantedit": "instant", "turboedit": "turbo", "swiftedit": "swift", "chord": "chord", "rrls": "rrls", "where": "where"}

def relative(value, label):
    from data import clean, locate
    return locate(Path("."), clean(value, label))

def fetch(root, roles):
    # Fetch only fixed public revisions, without touching existing snapshots.
    root = relative(root, "sources")
    if not roles or any(role not in RELEASES for role in roles):
        raise ValueError("Select known upstream source roles.")
    for role in dict.fromkeys(roles):
        repository, revision = RELEASES[role]
        target = root / role
        if target.exists():
            actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
            if actual != revision:
                raise ValueError("An existing upstream checkout has a different revision: " + role)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "https://github.com/" + repository + ".git", str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", revision], check=True)

def manifest(root, method):
    if method not in ROLES:
        raise ValueError("The method has no upstream source role.")
    role = ROLES[method]
    path = relative(root, "sources") / role
    if not path.is_dir():
        raise FileNotFoundError("Prepare the official source directory: " + str(path))
    files = sorted(path.rglob("*.py"))
    if not files:
        raise ValueError("The upstream source directory contains no Python implementation.")
    for item in files:
        if not item.resolve().is_relative_to(path.resolve()):
            raise ValueError("Upstream source links must stay inside their source directory.")
    repository, expected = RELEASES[role]
    actual = None
    if (path / ".git").exists():
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    return {"root": path.as_posix(), "repository": "https://github.com/" + repository, "expectedrevision": expected, "revision": actual, "files": {p.relative_to(path).as_posix(): digest(p) for p in files}}

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
