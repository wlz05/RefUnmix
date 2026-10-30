# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from collections import deque
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from copy import copy
import queue
from threading import Lock, Thread
from time import perf_counter
from traceback import clear_frames
import torch
from engine.judge import ClipJudge
from engine.session import Editor, number

def detached(error):
    # Keep stored failures free of tensor-bearing tracebacks.
    args = error.args if all(isinstance(value, (str, int, float, bool, type(None))) for value in error.args) else (str(error),)
    try:
        return type(error)(*args)
    except Exception:
        return RuntimeError(type(error).__name__ + ": " + str(error))

def options(args):
    width = getattr(args, "batch", 1)
    if type(width) is not int or not 1 <= width <= 4:
        raise ValueError("Batch width must be an integer from one to four.")
    if width > 1 and (getattr(args, "adapter", None) or args.method not in {"field", "path", "raw", "fixed"} or args.localization in {"attention", "hybrid"}):
        raise ValueError("Parallel requests require native field/path/raw/fixed modes without attention hooks; use --batch 1 for other modes.")
    if width > 1 and getattr(args, "deterministic", False):
        raise ValueError("Strict deterministic reproduction requires --batch 1.")
    return width

class Backend:
    parallel = True

    def __init__(self, shared, scheduler):
        self.shared, self.dispatcher, self.textcalls = shared, scheduler, 0

    def __getattr__(self, name):
        return getattr(self.shared, name)

    def prepare(self, *args, **kwargs):
        return Editor.prepare(self, *args, **kwargs)

    def prompt(self, value):
        result = self.dispatcher.call(self.shared, "prompt", value)
        pipeline = getattr(self.shared, "pipeline", None)
        self.textcalls += max(1, sum(getattr(pipeline, name, None) is not None for name in ("text_encoder", "text_encoder_2", "text_encoder_3")))
        return result

    def encode(self, value):
        return self.dispatcher.call(self.shared, "encode", value)

    def decode(self, value):
        return self.dispatcher.call(self.shared, "decode", value)

    def predict(self, *args):
        return self.dispatcher.call(self.shared, "predict", *args)

    def state(self, *args):
        return self.dispatcher.call(self.shared, "state", *args)

    def velocity(self, *args):
        return self.dispatcher.call(self.shared, "velocity", *args)

    def blend(self, *args):
        return self.dispatcher.call(self.shared, "blend", *args)

class Judge(ClipJudge):
    def __init__(self, shared, scheduler):
        self.__dict__.update(copy(shared.__dict__))
        self.shared, self.dispatcher = shared, scheduler
        self.evaluations, self.imagecalls, self.imagesamples = 0, 0, 0

    def images(self, values):
        result = self.dispatcher.call(self.shared, "images", values)
        self.imagecalls += 1
        self.imagesamples += len(values)
        return result

    def texts(self, values):
        return self.dispatcher.call(self.shared, "texts", values)

class Batch:
    def __init__(self, resources, width=3, window=.005):
        if type(width) is not int or not 2 <= width <= 4 or not 0 < number(window, "Batch collection window", 0, 1):
            raise ValueError("Parallel execution requires two to four requests and a positive collection window.")
        if len(resources) != 3:
            raise ValueError("Parallel execution requires backbone, control, and report resources.")
        self.resources, self.width, self.window = resources, width, window
        self.queue, self.pending, self.lock = queue.Queue(), deque(), Lock()
        self.failure, self.closed, self.started = None, False, False
        self.counts = {}
        self.active = ()

    def __enter__(self):
        if self.started or self.closed:
            raise RuntimeError("A batch scheduler cannot be reused.")
        self.started = True
        self.executor = ThreadPoolExecutor(max_workers=self.width, thread_name_prefix="request")
        self.thread = Thread(target=self.loop, name="inference", daemon=True)
        self.thread.start()
        return self

    def abort(self, error):
        with self.lock:
            if self.failure is None:
                self.failure = detached(error)

    def __exit__(self, kind, value, trace):
        if self.closed:
            return
        if kind is not None:
            self.abort(RuntimeError("Parallel execution stopped after a failed or interrupted request."))
        self.executor.shutdown(wait=True, cancel_futures=True)
        with self.lock:
            self.closed = True
        self.queue.put(None)
        self.thread.join()
        for future in self.active:
            if not future.cancelled() and future.done():
                error = future.exception()
                if error is not None:
                    clear_frames(error.__traceback__)
        self.active = ()
        self.pending.clear()

    def replicas(self):
        editor, judge, evaluation = self.resources
        return Backend(editor, self), Judge(judge, self) if judge is not None else None, Judge(evaluation, self) if evaluation is not None else None

    def map(self, function, values):
        if not self.started or self.closed or len(values) > self.width:
            raise ValueError("Submit at most one bounded request group to an active scheduler.")
        if self.failure is not None:
            raise detached(self.failure) from None
        futures = [self.executor.submit(function, value) for value in values]
        self.active = tuple(futures)
        try:
            done, _ = wait(futures, return_when=FIRST_EXCEPTION)
            for future in done:
                if future.exception() is not None:
                    future.result()
            results = [future.result() for future in futures]
            self.call(self, "barrier")
            self.active = ()
            return results
        except BaseException:
            self.abort(RuntimeError("Parallel execution stopped after a failed or interrupted request."))
            for future in futures:
                future.cancel()
            raise

    def call(self, owner, name, *args):
        with self.lock:
            if not self.started or self.closed:
                raise RuntimeError("The batch scheduler is not active.")
            if self.failure is not None:
                raise detached(self.failure) from None
            future = Future()
            self.queue.put((owner, name, args, future))
        return future.result()

    def key(self, request):
        owner, name, args, _ = request
        metadata = getattr(owner, "metadata", {})
        kinds = {"encode", "decode", "predict", "state"} if isinstance(metadata, dict) and metadata.get("family") == "vp" else {"encode", "decode"}
        if owner is not self.resources[0] or name not in kinds:
            return None
        tensors = [value for value in args if isinstance(value, torch.Tensor)]
        if not tensors or not all(value.ndim > 0 and value.shape[0] == tensors[0].shape[0] for value in tensors):
            return None
        return name, tuple((tuple(value.shape[1:]), value.device, value.dtype) if isinstance(value, torch.Tensor) else value for value in args)

    def loop(self):
        while True:
            first = self.pending.popleft() if self.pending else self.queue.get()
            if first is None:
                self.queue.task_done()
                return
            requests, current, key = [first], None, None
            try:
                key = self.key(first)
                deadline = perf_counter() + self.window
                while key is not None and len(requests) < self.width and perf_counter() < deadline and self.failure is None:
                    try:
                        current = self.queue.get(timeout=max(1e-6, deadline - perf_counter()))
                    except queue.Empty:
                        break
                    if current is None or self.key(current) != key:
                        self.pending.append(current)
                    else:
                        requests.append(current)
                self.dispatch(requests)
            except BaseException as error:
                self.abort(error)
                if current is not None and not any(value is current for value in requests) and not any(value is current for value in self.pending):
                    requests.append(current)
                for request in requests:
                    if not request[3].done():
                        request[3].set_exception(detached(error))
            finally:
                for request in requests:
                    self.queue.task_done()
                first = current = requests = request = key = None

    def dispatch(self, requests):
        owner, name, args, _ = requests[0]
        try:
            if self.failure is not None:
                raise detached(self.failure)
            if owner is self and name == "barrier":
                requests[0][3].set_result(None)
                return
            sizes = [request[2][0].shape[0] for request in requests] if self.key(requests[0]) is not None else None
            combined = tuple(torch.cat([request[2][index] for request in requests]) if isinstance(value, torch.Tensor) else value for index, value in enumerate(args)) if len(requests) > 1 else args
            row = self.counts.setdefault(name, {"requests": 0, "batches": 0, "samples": 0})
            row["requests"] += len(requests)
            row["batches"] += 1
            row["samples"] += sum(request[2][0].shape[0] if name in {"encode", "decode", "predict", "state", "velocity"} else len(request[2][0]) if name in {"images", "texts"} else 1 for request in requests)
            # Keep model execution on one inference thread.
            with torch.no_grad():
                result = getattr(owner, name)(*combined)
                if sizes is None:
                    outputs = [result]
                else:
                    if not isinstance(result, torch.Tensor) or result.shape[0] != sum(sizes):
                        raise RuntimeError("Batched model output has invalid geometry.")
                    outputs = [value.clone() for value in result.split(sizes)]
            for request, result in zip(requests, outputs):
                request[3].set_result(result)
        except BaseException as error:
            if isinstance(error, torch.cuda.OutOfMemoryError):
                error = RuntimeError("CUDA memory was exhausted. Completed rows are retained. Start a new output directory with --batch 1, or use --batch 2/3 after checking available memory; keep precision identical across comparisons.")
            self.abort(error)
            for request in requests:
                request[3].set_exception(detached(error).with_traceback(error.__traceback__))

    def report(self):
        totals = {key: sum(self.counts.get(name, {}).get(key, 0) for name in ("predict", "state", "velocity")) for key in ("samples", "requests", "batches")}
        return {"width": self.width, "sharedweights": True, "operations": {key: dict(value) for key, value in self.counts.items()}, "denoisersamples": totals["samples"], "denoiserrequests": totals["requests"], "denoisercalls": totals["batches"], "scope": "Physical shared model dispatches; per-request budgets retain their conditioned samples and logical forward calls."}

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
