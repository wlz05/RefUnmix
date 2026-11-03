# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import contextlib
from copy import deepcopy
import io
import json
import re
from pathlib import Path
import shlex
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
from run import common, parser, run, validate
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
        suites = ("tangent", "residual", "semantic", "alignment", "fields", "null", "core", "reference", "diagnosis", "leverage", "frontier", "attainment", "release", "geometry", "prior", "mechanism", "search", "guidance", "budget", "candidates", "cost", "solver", "goals")
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

    def testsemanticmatrix(self):
        args = self.args()
        args.suite = "semantic"
        plan = design(args, variants(args.suite, base=args))
        self.assertEqual(set(plan["variants"]["combined"]["changes"]), {"fieldrepair", "semantic"})
        for name in ("anchored", "regional", "repaired", "combined"):
            for key in ("maxcalls", "maxsamples", "candidates", "poststeps", "guidance"):
                self.assertEqual(plan["variants"][name]["settings"][key], getattr(args, key))
        args.goals = ["target"]
        with self.assertRaises(ValueError):
            design(args, variants(args.suite, base=args))

    def testtangentmatrix(self):
        args = self.args()
        args.suite = "tangent"
        plan = design(args, variants(args.suite, base=args))
        self.assertFalse(plan["variants"]["full"]["changes"])
        self.assertEqual(set(plan["variants"]), {"full", "anchorraw", "tangentraw", "matchedraw", "temporalraw", "controlled"})
        for name, value in plan["variants"].items():
            self.assertEqual(value["settings"]["maxsamples"], args.maxsamples)
            self.assertEqual(value["settings"]["poststeps"], 1)

    def testtangentruntime(self):
        image = Image.new("RGB", (16, 16), (90, 128, 160))
        for method in ("field", "raw", "path"):
            args = self.args("--tangent", "--method", method)
            output, report, state = run(editor(), ClipJudge(Clip(), Processor()), image, "source", "target long", args)
            self.assertTrue(report["controlactive"])
            self.assertIn("tangent", report["field"])
            self.assertLessEqual(state.samples, args.maxsamples)
            self.assertEqual(state.samples, state.limit.used)
            self.assertEqual(output.shape, state.origin.shape)
            if method == "path":
                self.assertEqual(report["path"]["rawreference"], "initial field before repair")
            json.dumps(report, allow_nan=False)

    def testintegratedruntime(self):
        args = self.args("--method", "field", "--semantic", "region")
        image = Image.new("RGB", (16, 16), (90, 128, 160))
        with patch("engine.poll.select", side_effect=AssertionError("Separate control is disabled.")), patch("engine.path.Path", side_effect=AssertionError("Reference paths are disabled.")):
            _, report, session = run(editor(), ClipJudge(Clip(), Processor()), image, "source", "target long", args)
        self.assertFalse(report["controller"]["enabled"])
        self.assertEqual(report["candidates"], [])
        trace = report["field"]["tangent"]
        self.assertEqual(report["evaluations"], trace["evaluations"])
        self.assertEqual(report["judgestats"]["evaluations"], trace["evaluations"])
        self.assertEqual(session.samples, session.limit.used)
        self.assertEqual(session.calls, session.limit.forwards)
        self.assertLessEqual(report["damage"], report["rawdamage"])
        self.assertNotIn("path", report)

    def testintegratedmatrix(self):
        args = self.args("--method", "field", "--semantic", "region")
        args.suite = "integrated"
        plan = design(args, variants(args.suite, base=args))
        self.assertFalse(plan["variants"]["full"]["changes"])
        self.assertEqual(set(plan["variants"]), {"full", "anchor", "temporal", "matched", "legacy", "controlled"})
        for name, row in plan["variants"].items():
            self.assertEqual(row["settings"]["poststeps"], 1)
            self.assertEqual(row["settings"]["maxsamples"], args.maxsamples)
        args.suite = "fields"
        with self.assertRaisesRegex(ValueError, "suite integrated"):
            design(args, variants(args.suite, base=args))

    def testintegratedflags(self):
        for flags in (("--prior", "rec"), ("--control",), ("--compensation",), ("--frontier",), ("--field", "chord"), ("--fieldrepair", "1")):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                validate(self.args("--method", "field", *flags))

    def testintegratedequivalence(self):
        image = Image.new("RGB", (16, 16), (90, 128, 160))
        left = self.args("--method", "field", "--semantic", "region")
        right = self.args("--method", "raw", "--tangent", "--semantic", "region")
        first = run(editor(), ClipJudge(Clip(), Processor()), image, "source", "target long", left)
        second = run(editor(), ClipJudge(Clip(), Processor()), image, "source", "target long", right)
        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
        for key in ("calls", "samples", "decodes"):
            self.assertEqual(first[1]["stats"][key], second[1]["stats"][key])
        self.assertEqual(first[1]["scores"], second[1]["scores"])

    def testintegratedgeometry(self):
        image = Image.new("RGB", (24, 24), (90, 128, 160))
        args = self.args("--method", "field", "--size", "24")
        backend = editor()
        backend.size = 24
        output, report, _ = run(backend, ClipJudge(Clip(), Processor()), image, "source", "target long", args)
        self.assertEqual(tuple(output.shape[-2:]), (24, 24))
        self.assertEqual(report["method"], "field")
        with self.assertRaisesRegex(ValueError, "even latent"):
            validate(self.args("--size", "24"))

    def testamplitudematrix(self):
        args = self.args("--method", "field", "--semantic", "region")
        args.suite = "amplitude"
        plan = design(args, variants(args.suite, base=args))
        self.assertEqual(len(plan["variants"]), 10)
        self.assertFalse(plan["variants"]["full"]["changes"])
        self.assertTrue(plan["variants"]["feedback"]["settings"]["tangentfeedback"])
        for name, factor in (("zero", 0.), ("half", .5), ("weak", .75), ("anchor", 1.), ("strong", 1.25), ("large", 1.5), ("double", 2.)):
            settings = plan["variants"][name]["settings"]
            self.assertEqual(settings["scale"], args.scale)
            self.assertEqual(settings["fieldfactor"], factor)
            self.assertEqual(settings["method"], "raw")

    def testenvelopematrix(self):
        args = self.args("--method", "field", "--semantic", "region", "--tangentbasis", "shared", "--tangentpolicy", "frontier")
        args.suite = "envelope"
        plan = design(args, variants(args.suite, base=args))
        self.assertEqual(len(plan["variants"]), 7)
        self.assertFalse(plan["variants"]["full"]["changes"])
        self.assertEqual(set(plan["variants"]["conditional"]["changes"]), {"tangentpolicy"})
        self.assertEqual(set(plan["variants"]["residual"]["changes"]), {"tangentbasis"})

    def testenveloperuntime(self):
        args = self.args("--method", "field", "--semantic", "region", "--tangentbasis", "shared", "--tangentpolicy", "frontier")
        image = Image.new("RGB", (16, 16), (90, 128, 160))
        _, report, state = run(editor(), ClipJudge(Clip(), Processor()), image, "source", "target long", args)
        trace = report["field"]["tangent"]
        self.assertEqual(trace["policy"], "frontier")
        self.assertEqual(trace["dictionarydim"], trace["rank"])
        self.assertEqual(trace["tangentdim"], trace["dictionarydim"] - 1)
        self.assertEqual(trace["semanticheads"], len(trace["heads"]))
        self.assertEqual(trace["probes"], trace["tangentdim"])
        self.assertTrue(trace["dimensionsmeasured"])
        self.assertLessEqual(trace["checks"], 3)
        self.assertEqual(state.samples, state.limit.used)
        self.assertEqual(state.calls, state.limit.forwards)
        self.assertLessEqual(report["damage"], report["rawdamage"])
        json.dumps(report, allow_nan=False)

    def testconditionalmatrix(self):
        args = self.args("--method", "field", "--semantic", "region", "--tangentbasis", "shared", "--tangentpolicy", "conditional")
        args.suite = "conditional"
        plan = design(args, variants(args.suite, base=args))
        self.assertEqual(len(plan["variants"]), 7)
        self.assertFalse(plan["variants"]["full"]["changes"])
        self.assertEqual(set(plan["variants"]["expanded"]["changes"]), {"tangentpolicy"})
        self.assertEqual(set(plan["variants"]["residual"]["changes"]), {"tangentbasis"})
        for name in ("full", "expanded", "feedback", "matched"):
            self.assertEqual(plan["variants"][name]["settings"]["tangentbasis"], "shared")

    def testconditionalruntime(self):
        args = self.args("--method", "field", "--semantic", "region", "--tangentbasis", "shared", "--tangentpolicy", "conditional")
        image = Image.new("RGB", (16, 16), (90, 128, 160))
        _, report, state = run(editor(), ClipJudge(Clip(), Processor()), image, "source", "target long", args)
        trace = report["field"]["tangent"]
        self.assertEqual(trace["policy"], "conditional")
        self.assertEqual(state.samples, state.limit.used)
        self.assertEqual(state.calls, state.limit.forwards)
        self.assertLessEqual(report["damage"], report["rawdamage"])
        self.assertEqual(trace["sources"], 1)
        json.dumps(report, allow_nan=False)

    def testconditionalflags(self):
        for flags in (("--method", "raw"), ("--method", "fixed")):
            with self.assertRaises(ValueError):
                validate(self.args("--tangentpolicy", "conditional", *flags))
        args = self.args("--method", "field", "--semantic", "region", "--tangentpolicy", "conditional")
        args.suite = "conditional"
        with self.assertRaisesRegex(ValueError, "shared coupled"):
            design(args, variants(args.suite, base=args))

    def testconditionalreferencecache(self):
        from engine.native import Native
        original, calls = Native.endpoint, []
        def endpoint(native, field, eta=0.):
            if eta == 1 and not field.ne(0).any().item():
                calls.append(eta)
            return original(native, field, eta)
        args = self.args("--method", "path", "--tangent", "--tangentbasis", "shared", "--tangentpolicy", "conditional")
        image = Image.new("RGB", (16, 16), (90, 128, 160))
        with patch.object(Native, "endpoint", endpoint):
            _, report, state = run(editor(), ClipJudge(Clip(), Processor()), image, "source", "target long", args)
        self.assertEqual(calls, [1.])
        self.assertEqual(state.samples, state.limit.used)

    def testamplitudefields(self):
        image = Image.new("RGB", (16, 16), (90, 128, 160))
        hashes = []
        sizes = []
        for factor in (0., .5, .75, 1., 1.25, 1.5, 2.):
            args = self.args("--method", "raw", "--fieldfactor", str(factor))
            _, report, session = run(editor(), None, image, "source", "target long", args)
            hashes.append(report["anchorfieldsha256"])
            sizes.append(report["fieldnorm"])
            self.assertEqual(session.samples, session.limit.used)
            self.assertIn("responsenorm", report["field"])
        self.assertEqual(len(set(hashes)), 1)
        for value, factor in zip(sizes, (0., .5, .75, 1., 1.25, 1.5, 2.)):
            self.assertAlmostEqual(value, factor * sizes[3], places=6)

    def testfeedbackruntime(self):
        args = self.args("--method", "field", "--semantic", "region", "--tangentfeedback")
        image = Image.new("RGB", (16, 16), (90, 128, 160))
        _, report, state = run(editor(), ClipJudge(Clip(), Processor()), image, "source", "target long", args)
        self.assertEqual(state.samples, state.limit.used)
        self.assertFalse(report["controller"]["enabled"])
        self.assertLessEqual(report["field"]["tangent"]["checks"], 3)

    def testamplitudeflags(self):
        for flags in (("--method", "field"), ("--tangent",), ("--fieldrepair", "1"), ("--field", "chord"), ("--fieldfactor", "nan"), ("--fieldfactor", "-1")):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                validate(self.args("--method", "raw", "--fieldfactor", "1.5", *flags))

    def testsemanticruntime(self):
        args = self.args("--semantic", "region", "--fieldrepair", "1")
        image = Image.new("RGB", (16, 16), (90, 128, 160))
        _, report, session = run(editor(), ClipJudge(Clip(), Processor()), image, "source", "target long", args)
        self.assertEqual(report["semanticviews"]["layout"], "region")
        self.assertLessEqual(session.samples, args.maxsamples)
        self.assertEqual(session.samples, session.limit.used)
        self.assertNotIn("view0", report["scores"])
        self.assertIn("repair", report["field"])
        original = deepcopy(args)
        original.fieldrepair = 0
        _, reference, _ = run(editor(), ClipJudge(Clip(), Processor()), image, "source", "target long", original)
        self.assertEqual(report["rawdamage"], reference["rawdamage"])
        self.assertEqual(report["rawscores"], reference["rawscores"])
        self.assertLessEqual(report["damage"], reference["rawdamage"])
        self.assertEqual(report["path"]["rawreference"], "initial field before repair")

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

    def testrecipepaths(self):
        from trial import settings
        first, second = self.args(), self.args()
        first.model, first.clip, first.evalclip = ".\\models", ".\\clip", ".\\report"
        second.model, second.clip, second.evalclip = "models", "clip", "report"
        self.assertEqual(settings(first), settings(second))

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

    def testfieldbenchmarkplans(self):
        for suite, count in (("integrated", 6), ("amplitude", 10), ("conditional", 7)):
            arguments = ["trial.py", "--suite", suite, "--split", "all", "--method", "field", "--semantic", "region", "--model", "missing", "--clip", "missing", "--dry"]
            if suite == "conditional":
                arguments += ["--tangentbasis", "shared", "--tangentpolicy", "conditional"]
            stream = io.StringIO()
            with self.subTest(suite=suite), patch("sys.argv", arguments), patch("trial.items") as items, patch("trial.load") as load, contextlib.redirect_stdout(stream):
                main()
            self.assertEqual(len(json.loads(stream.getvalue())["design"]["variants"]), count)
            items.assert_not_called()
            load.assert_not_called()

    def documented(self, extra=(), profile="main"):
        from conduct import parser as reproduction, generation
        root = Path(__file__).resolve().parent.parent
        text = (root / "README.md").read_text(encoding="utf-8")
        args = reproduction().parse_args(["--profile", profile])
        command = generation(args, 42, Path("outputs") / profile / "42")
        options = datasetargs(common(argparse.ArgumentParser())).parse_args(command[3:] + list(extra))
        validate(options)
        return text, options

    def testdocumentedexecution(self):
        from conduct import parser as reproduction
        text, _ = self.documented()
        commands = [shlex.split(line) for line in text.splitlines() if line.startswith("python -B conduct.py --seeds")]
        seeds = [reproduction().parse_args(command[3:]).seeds for command in commands]
        self.assertEqual(seeds, [[42], [52], [3047], [3407]])
        self.assertNotIn("experiments/single.py", text)

    def testbenchmarkpartitions(self):
        for options in (("--manifest", "split.json"), ("--recipe", "recipe.json")):
            arguments = ["trial.py", "--split", "all", "--clip", "clip", "--dry", *options]
            with self.subTest(options=options), patch("sys.argv", arguments), patch("trial.load") as load:
                with self.assertRaisesRegex(ValueError, "no manifest or validation recipe"):
                    main()
            load.assert_not_called()

    def testdocumentedplans(self):
        _, options = self.documented()
        self.assertTrue(options.single)
        self.assertEqual((options.method, options.singlemode, options.singleend), ("raw", "recenter", "responsibility"))
        self.assertEqual((options.start, options.delta, options.end), (.75, 0., .30))
        self.assertEqual((options.recenter, options.conditiongain, options.innovationgain), (.5, 1.25, 2.))
        self.assertEqual((options.batch, options.precision, options.device), (3, "fp32", "cuda:0"))
        self.assertFalse(options.control or options.heldout or options.compensation or options.frontier or options.cfg)
        self.assertEqual((options.poststeps, options.prior), (1, "off"))
        self.assertEqual((options.mapping, options.images, options.split), ("mapping_file.json", "annotation_images", "all"))

    def testdocumentedcurves(self):
        text, _ = self.documented()
        commands = [shlex.split(line) for line in text.splitlines() if line.startswith("python -B conduct.py --profile")]
        self.assertEqual(len(commands), 5)
        profiles = [self.documented(profile=command[command.index("--profile") + 1])[1] for command in commands]
        self.assertTrue(any(value.innovationgain == 1 for value in profiles))
        self.assertTrue(any(value.recenter == 0 for value in profiles))
        self.assertTrue(any(value.conditiongain == 1 and value.singlegain == 1.5 for value in profiles))
        self.assertTrue(any(not value.singlebias for value in profiles))
        self.assertTrue(any(value.biasgeometry == "global" for value in profiles))

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

    def testdocumentedassessment(self):
        from assess import parser as assessment
        from conduct import parser as reproduction, scoring
        text, _ = self.documented()
        command = scoring(reproduction().parse_args([]), 42, Path("outputs/main/42"))
        options = assessment().parse_args(command[3:])
        self.assertEqual((options.scope, options.seeds), ("benchmark", [42]))
        self.assertEqual(options.protocol, "protocol/metrics.py")
        self.assertTrue(options.offline)
        self.assertFalse(options.network or options.pixels)
        self.assertEqual((options.cache, options.device), ("cache/metrics", "cuda:0"))
        fence = chr(96) * 3
        blocks = re.findall(fence + r"bash\n(.*?)" + fence, text, re.S)
        aggregation = next(block for block in blocks if "aggregate.py" in block and "--input" in block)
        command = shlex.split(aggregation.replace("\\\n", " "))
        index = command.index("--seeds")
        self.assertEqual(list(map(int, command[index + 1:])), [42, 52, 3047, 3407])
        inputs = command[command.index("--input") + 1:index]
        self.assertEqual(len(set(inputs)), 4)
        self.assertTrue(all(not Path(path).is_absolute() for path in inputs))

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

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
