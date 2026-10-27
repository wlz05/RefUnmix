# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import contextlib
from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from audit import agreement
from engine.activity import prior as activity
from engine.attainment import choose
from engine.judge import ClipJudge
from engine.poll import Candidate, Config, select
from engine.protocol import configuration, signature
from evaluate import coverage, datasetargs
from run import common, parser, run
from trial import design, frozen, main, settings, variants
from session import Clip, Processor, editor
from trust import Expensive, Fake

class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def args(self, *values):
        args = datasetargs(common(argparse.ArgumentParser())).parse_args(["--clip", "clip", "--size", "16", *values])
        args.suite, args.recipe, args.dry = "fields", None, False
        return args

    def candidate(self, key, damage=1., gain=0., penalty=0., prior=0., **values):
        pixels = torch.zeros(1, 3, 4, 4)
        return Candidate(key, None, pixels, pixels, {"target": gain}, damage, prior=prior, feasible=True, complete=True, penalty=penalty, objective=damage + penalty, taskgain=gain, **values)

    def testinheritance(self):
        args = self.args("--prior", "rec", "--compensation", "--no-release", "--poststeps", "2", "--cfg", "--guidance", "2", "--referencepath", "prediction")
        plan = design(args, variants("fields", base=args))
        self.assertEqual(plan["main"], "pullbackfull")
        self.assertEqual(plan["variants"]["pullbackfull"]["sha256"], signature(args)["sha256"])
        self.assertEqual(plan["variants"]["pullbackfull"]["changes"], {})
        for key in ("prior", "compensation", "release", "poststeps", "cfg", "guidance", "referencepath"):
            self.assertEqual(plan["variants"]["pullbackfull"]["settings"][key], getattr(args, key))

    def testallsuites(self):
        suites = ("alignment", "fields", "null", "core", "reference", "diagnosis", "leverage", "frontier", "attainment", "release", "geometry", "prior", "mechanism", "search", "guidance", "budget", "candidates", "cost", "solver", "goals")
        for suite in suites:
            args = self.args()
            args.suite = suite
            plan = design(args, variants(suite, base=args))
            self.assertEqual(plan["variants"][plan["main"]]["changes"], {})

    def testfactorchanges(self):
        args = self.args()
        args.suite = "release"
        plan = design(args, variants(args.suite, base=args))
        self.assertEqual(set(plan["variants"]["joint"]["changes"]), {"release"})
        args.suite = "leverage"
        plan = design(args, variants(args.suite, base=args))
        self.assertEqual(set(plan["variants"]["compensated"]["changes"]), {"compensation"})

    def testmismatch(self):
        args = self.args()
        controls = variants("fields", base=args)
        controls["pullbackfull"]["poststeps"] = 5
        with self.assertRaisesRegex(ValueError, "poststeps"):
            design(args, controls)

    def testnormalizedtimes(self):
        args = self.args("--times", "0.90", "0.60", "0.75", "0.90")
        plan = design(args, variants("fields", base=args))
        self.assertEqual(plan["variants"]["pullbackfull"]["changes"], {})
        self.assertEqual(args.times, [.60, .75, .90])

    def testidentityscope(self):
        args = self.args()
        other = deepcopy(args)
        other.output, other.split, other.seed, other.recipe, other.dry = "different", "val", 52, "frozen.json", True
        self.assertEqual(signature(args), signature(other))
        other.maxsamples = 15
        self.assertNotEqual(signature(args)["sha256"], signature(other)["sha256"])
        other = deepcopy(args)
        other.model = ".\\models"
        self.assertEqual(configuration(args), configuration(other))

    def testentryidentity(self):
        single = parser().parse_args(["--image", "input.png", "--source", "source", "--target", "target", "--clip", "clip", "--size", "16"])
        self.assertEqual(signature(single), signature(self.args()))

    def testdry(self):
        stream = io.StringIO()
        arguments = ["trial.py", "--suite", "fields", "--manifest", "missing.json", "--model", "missing", "--clip", "missing", "--dry"]
        with patch("sys.argv", arguments), patch("trial.items") as items, patch("trial.load") as load, contextlib.redirect_stdout(stream):
            main()
        items.assert_not_called()
        load.assert_not_called()
        plan = json.loads(stream.getvalue())["design"]
        self.assertEqual(len(plan["variants"]), 12)
        self.assertEqual(plan["variants"]["pullbackfull"]["changes"], {})

    def testtamperedrecipe(self):
        args = self.args()
        args.suite, args.recipe, args.manifest = "core", "recipe.json", "split.json"
        payload = {"schema": 1, "split": "val", "suite": "core", "settings": settings(args), "variants": variants("core", base=args), "digests": {}, "manifestdigest": "manifest", "mappingdigest": "manifest"}
        payload["variants"]["full"]["prior"] = "rec"
        with patch("trial.locate"), patch("trial.read", return_value=payload), patch("trial.identities", return_value={}), patch("trial.digest", return_value="manifest"):
            with self.assertRaisesRegex(ValueError, "declared ablation"):
                frozen(args, [])

    def testbenchmarkplan(self):
        stream = io.StringIO()
        arguments = ["trial.py", "--suite", "fields", "--split", "all", "--model", "missing", "--clip", "missing", "--dry"]
        with patch("sys.argv", arguments), patch("trial.items") as items, patch("trial.load") as load, contextlib.redirect_stdout(stream):
            main()
        items.assert_not_called()
        load.assert_not_called()
        self.assertEqual(len(json.loads(stream.getvalue())["design"]["variants"]), 12)

    def testbenchmarkpartitions(self):
        for options in (("--manifest", "split.json"), ("--recipe", "recipe.json")):
            arguments = ["trial.py", "--split", "all", "--clip", "clip", "--dry", *options]
            with self.subTest(options=options), patch("sys.argv", arguments), patch("trial.load") as load:
                with self.assertRaisesRegex(ValueError, "no manifest or validation recipe"):
                    main()
            load.assert_not_called()

    def testrenderidentity(self):
        first = self.args()
        second = deepcopy(first)
        for key, value in variants("fields", base=first)["pullbackfull"].items():
            setattr(second, key, value)
        image = Image.new("RGB", (16, 16), (96, 128, 160))
        outputs = [run(editor(), ClipJudge(Clip(), Processor()), image, "source", "target long", args, Image.new("L", image.size, 128)) for args in (first, second)]
        self.assertTrue(torch.equal(outputs[0][0], outputs[1][0]))
        for key in ("controlsignature", "fieldsha256", "point", "scores", "damage"):
            self.assertEqual(outputs[0][1][key], outputs[1][1][key])

    def testuniquewinner(self):
        raw = self.candidate("raw")
        first = self.candidate("0", .2, 2., .09, 2.)
        second = self.candidate("1", .1, 1., 0., 0.)
        selected, _ = choose([raw, first, second])
        record = activity(raw, [first, second], selected, Config(), {"evaluations": 3}, "ddpm")
        self.assertFalse(record["selectionchanged"])
        self.assertEqual(record["reason"], "attainment band has one candidate")
        self.assertEqual(record["penalized"], 1)

    def testpriorchangeswinner(self):
        raw = self.candidate("raw")
        first = self.candidate("0", .2, 1., .09, 2.)
        second = self.candidate("1", .25, 1., 0., 0.)
        selected, _ = choose([raw, first, second])
        record = activity(raw, [first, second], selected, Config(), {"evaluations": 3}, "ddpm")
        self.assertTrue(record["selectionchanged"])
        self.assertEqual((record["selectedwithprior"], record["selectedwithoutprior"]), ("1", "0"))

    def testhardveto(self):
        raw = self.candidate("raw")
        first = self.candidate("0", .2, 2., reasons=("prior",))
        first.feasible = False
        second = self.candidate("1", .25, 1.)
        selected, _ = choose([raw, first, second])
        record = activity(raw, [first, second], selected, Config(priorcontrol="hard"), {"evaluations": 3}, "ddpm")
        self.assertTrue(record["selectionchanged"])
        self.assertEqual(record["vetoed"], 1)

    def testprioroff(self):
        raw = self.candidate("raw", prior=None)
        record = activity(raw, [], raw, Config(), {}, "off")
        self.assertEqual(record["reason"], "disabled")
        self.assertFalse(record["enabled"])

    def testpriorserialization(self):
        raw = self.candidate("raw", prior=np.float64(0.))
        first = self.candidate("0", .2, 1., np.float64(.09), np.float64(2.))
        selected, _ = choose([raw, first])
        record = activity(raw, [first], selected, Config(), {"evaluations": 2}, "ddpm")
        self.assertIs(type(record["positiveexcess"]), int)
        self.assertIs(type(record["penalized"]), int)
        json.dumps(record, allow_nan=False)

    def testrescuetrace(self):
        path = Fake()
        report = select(path, lambda image: {"target": 1.}, torch.ones_like(path.session.origin[:, :1]), Expensive(), Config(compensation=True)).report
        self.assertGreater(report["rescue"]["considered"], 0)
        self.assertEqual(report["rescue"]["considered"], report["rescue"]["proposed"] + sum(report["rescue"]["skipped"].values()))
        self.assertEqual(report["rescue"]["queries"], sum(row.get("kind") == "compensation" for row in report["history"]))

    def testactivitycoverage(self):
        record = {"method": "path", "prioractivity": {"enabled": True, "evaluations": 3, "penalized": 2, "selectionchanged": False, "reason": "same-pool selection unchanged"}, "rescue": {"eligible": 1, "selected": True}}
        result = coverage([{"report": record}, {"report": {"method": "path"}}])
        self.assertEqual(result["prior"]["missing"], 1)
        self.assertEqual(result["prior"]["penalized"], 1)
        self.assertEqual(result["prior"]["selectionchanged"], 0)
        self.assertEqual(result["rescueactivity"]["selected"], 1)

    def testoutputagreement(self):
        left = [{"key": "a", "seed": 42, "resultsha256": "a" * 64, "report": {"candidates": [{"point": [1., 1., 0.]}]}}, {"key": "b", "seed": 42}]
        right = deepcopy(left)
        result = agreement(left, right)
        self.assertEqual((result["identical"], result["missinghash"], result["samecandidatepath"]), (1, 1, 1))
        right[0]["resultsha256"] = "b" * 64
        self.assertEqual(agreement(left, right)["different"], 1)
        with self.assertRaises(ValueError):
            agreement(left, right[:1])
        with self.assertRaises(ValueError):
            agreement(left + left, right + right)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
