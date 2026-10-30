# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import contextlib
from copy import deepcopy
import io
import json
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

    def testdocumentedexecution(self):
        lines = Path(__file__).resolve().parent.parent.joinpath("README.md").read_text(encoding="utf-8").splitlines()
        commands = [shlex.split(line) for line in lines if line.startswith("python -B RefUnmix/trial.py") and "--split all" in line and "--dry" not in line]
        self.assertEqual(len(commands), 29)
        for command in commands:
            arguments = command[2:]
            if "--output" in arguments:
                index = arguments.index("--output")
                arguments = arguments[:index] + arguments[index + 2:]
            arguments += ["--dry"]
            stream = io.StringIO()
            with self.subTest(suite=arguments[arguments.index("--suite") + 1]), patch("sys.argv", arguments), patch("trial.items") as items, patch("trial.load") as load, contextlib.redirect_stdout(stream):
                main()
            payload = json.loads(stream.getvalue())
            self.assertFalse(payload["design"]["variants"][payload["design"]["main"]]["changes"])
            items.assert_not_called()
            load.assert_not_called()

    def testbenchmarkpartitions(self):
        for options in (("--manifest", "split.json"), ("--recipe", "recipe.json")):
            arguments = ["trial.py", "--split", "all", "--clip", "clip", "--dry", *options]
            with self.subTest(options=options), patch("sys.argv", arguments), patch("trial.load") as load:
                with self.assertRaisesRegex(ValueError, "no manifest or validation recipe"):
                    main()
            load.assert_not_called()

    def testdocumentedplans(self):
        from evaluate import main as evaluatemain
        from trial import settings
        original = argparse.ArgumentParser.parse_args
        parsed = []

        class Parsed(BaseException):
            pass

        def capture(parser, *values, **options):
            parsed.append(original(parser, *values, **options))
            raise Parsed()

        lines = Path(__file__).resolve().parent.parent.joinpath("README.md").read_text(encoding="utf-8").splitlines()
        commands = [shlex.split(line) for line in lines if line.startswith(("python -B RefUnmix/evaluate.py", "python -B RefUnmix/trial.py"))]
        executions = []
        for command in commands:
            entry = evaluatemain if Path(command[2]).stem == "evaluate" else main
            with patch("sys.argv", command[2:]), patch.object(argparse.ArgumentParser, "parse_args", capture), self.assertRaises(Parsed):
                entry()
            executions.append((entry, parsed[-1]))
        reference = next(args for entry, args in executions if entry is evaluatemain and args.maxsamples == 64 and args.method == "path")
        integrated = next(args for entry, args in executions if entry is evaluatemain and args.maxsamples == 64 and args.method == "field" and args.tangentbasis == "residual")
        validate(reference)
        benchmark = [args for entry, args in executions if entry is main and args.split == "all" and not args.dry]
        self.assertEqual(len(benchmark), 29)
        self.assertEqual(sum(len(variants(args.suite, base=args)) for args in benchmark), 246)
        for args in benchmark:
            plan = design(args, variants(args.suite, base=args))
            expected = deepcopy(integrated if args.suite in {"integrated", "amplitude", "conditional"} else reference)
            if args.suite == "mechanism":
                expected.maxsamples = expected.maxcalls = 256
            elif args.suite in {"amplitude", "conditional"}:
                expected.maxsamples, expected.maxcalls = args.maxsamples, args.maxcalls
                if args.suite == "conditional":
                    expected.tangentbasis, expected.tangentpolicy = "shared", "conditional"
            self.assertEqual(plan["controlsignature"], signature(expected))
            self.assertEqual((args.batch, args.device, args.precision, args.seeds), (3, "cuda:0", "fp32", [42, 52, 3047, 3407]))
            self.assertIsNone(args.manifest)
            self.assertIsNone(args.recipe)
        validation = next(args for entry, args in executions if entry is main and args.split == "val")
        testing = next(args for entry, args in executions if entry is main and args.split == "test")
        self.assertEqual(settings(validation), settings(testing))
        self.assertEqual(Path(testing.recipe), Path(validation.output) / "recipe.json")

    def testdocumentedcurves(self):
        from assess import parser as assessment
        from audit import main as auditmain
        from engine.curve import CONDITIONAL, RADIAL
        lines = Path(__file__).resolve().parent.parent.joinpath("README.md").read_text(encoding="utf-8").splitlines()
        commands = [shlex.split(line) for line in lines if any(line.startswith("python -B RefUnmix/assess.py --input outputs/" + name + "/") for name in ("amplitude", "amplitudefifteen", "conditional", "conditionalfifteen"))]
        self.assertEqual(len(commands), 34)
        groups = {}
        for command in commands:
            args = assessment().parse_args(command[3:])
            label = Path(args.input).parent.name
            group = Path(args.input).parts[1]
            groups.setdefault(group, set()).add(label)
            self.assertEqual(Path(args.output).stem, label)
            self.assertEqual(args.seeds, [42, 52, 3047, 3407])
            self.assertEqual(args.scope, "benchmark")
            self.assertEqual(args.protocol, "protocol/metrics.py")
            self.assertEqual(args.cache, "cache/metrics")
            self.assertTrue(args.network)
            self.assertFalse(Path(args.input).is_absolute())
        expected = set(RADIAL) | {"full", "matched", "feedback"}
        self.assertEqual(groups, {"amplitude": expected, "amplitudefifteen": expected, "conditional": CONDITIONAL, "conditionalfifteen": CONDITIONAL})
        original = argparse.ArgumentParser.parse_args
        parsed = []
        class Parsed(BaseException):
            pass
        def capture(parser, *values, **options):
            parsed.append(original(parser, *values, **options))
            raise Parsed()
        commands = [shlex.split(line) for line in lines if line.startswith("python -B RefUnmix/audit.py") and "--sweep" in line]
        self.assertEqual(len(commands), 4)
        for command in commands:
            with patch("sys.argv", command[2:]), patch.object(argparse.ArgumentParser, "parse_args", capture), self.assertRaises(Parsed):
                auditmain()
        self.assertEqual({args.sweep for args in parsed}, {"outputs/amplitudemetrics", "outputs/amplitudemetricsfifteen", "outputs/conditionalmetrics", "outputs/conditionalmetricsfifteen"})

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
        lines = Path(__file__).resolve().parent.parent.joinpath("README.md").read_text(encoding="utf-8").splitlines()
        commands = [shlex.split(line) for line in lines if line.startswith("python -B RefUnmix/assess.py")]
        self.assertGreaterEqual(len(commands), 5)
        for command in commands:
            args = assessment().parse_args(command[3:])
            with self.subTest(input=args.input):
                if args.scope == "benchmark":
                    self.assertEqual(args.seeds, [42, 52, 3047, 3407])
                    self.assertEqual(args.protocol, "protocol/metrics.py")
                    self.assertEqual(args.device, "cuda:0")
                    self.assertNotEqual(args.network, args.offline)
                    self.assertFalse(args.pixels)
                else:
                    self.assertTrue(args.pixels)
                    self.assertEqual(args.seeds, [42])
                    self.assertFalse(args.network or args.offline)

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
