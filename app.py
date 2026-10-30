# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import argparse
from copy import deepcopy
from run import common, load, run

def port(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return number

def arguments() -> argparse.Namespace:
    parser = common(argparse.ArgumentParser(description="Launch native reference path editing."))
    parser.add_argument("--port", type=port, default=7860)
    return parser.parse_args()

def main() -> None:
    args = arguments()
    import gradio as gr
    from engine.session import toimage
    editor, judge = load(args)

    def edit(image, source, target, region, polarity, method, field, seed, noise, candidates, samples, axes, eta, representation, prior, steps, localtext):
        if image is None or not source.strip() or not target.strip():
            raise gr.Error("Image and prompts are required.")
        current = deepcopy(args)
        for key, value in {"method": method, "field": field, "seed": int(seed), "noise": int(noise), "candidates": int(candidates), "maxsamples": int(samples), "axes": axes, "eta": float(eta), "representation": representation, "prior": prior, "poststeps": int(steps), "localtext": localtext.strip() or None}.items():
            setattr(current, key, value)
        if method == "fixed":
            current.axes, current.prior, current.heldout = "none", "off", False
        try:
            pixels, report, _ = run(editor, judge, image, source.strip(), target.strip(), current, region, polarity)
        except (ValueError, RuntimeError) as error:
            raise gr.Error(str(error)) from error
        return toimage(pixels), report

    with gr.Blocks(title="RefUnmix") as demo:
        gr.Markdown("# RefUnmix")
        with gr.Row():
            image = gr.Image(type="pil", label="Image")
            output = gr.Image(type="pil", label="Result")
        source = gr.Textbox(label="Source")
        target = gr.Textbox(label="Target")
        method = gr.Dropdown(choices=("field", "path", "raw", "fixed"), value=args.method if args.method in {"field", "path", "raw", "fixed"} else "path", label="Execution")
        field = gr.Dropdown(choices=("pullback", "chord", "instant", "average"), value=args.field, label="Input field")
        region = gr.Image(type="pil", image_mode="L", label="Optional region mask")
        polarity = gr.Radio(choices=("edit", "protect"), value="edit", label="White mask pixels")
        localtext = gr.Textbox(label="Optional local target description")
        report = gr.JSON(label="Measured report")
        with gr.Row():
            seed = gr.Number(value=42, precision=0, label="Seed")
            noise = gr.Slider(1, 4, value=1, step=1, label="Noise")
            candidates = gr.Slider(1, 64, value=args.candidates, step=1, label="Candidate budget")
            samples = gr.Slider(8, 512, value=args.maxsamples, step=1, label="Model sample cap")
        with gr.Row():
            axes = gr.Dropdown(choices=("all", "path", "field", "none"), value=args.axes, label="Active coordinates")
            eta = gr.Slider(0, 1, value=args.eta, step=.05, label="Fixed reference coefficient")
            representation = gr.Dropdown(choices=("latent", "rgb", "mix", "random", "cfg"), value=args.representation, label="Reference control")
            prior = gr.Dropdown(choices=("ddpm", "rec", "off") if args.backbone == "vp" else ("flow", "off"), value=args.prior, label="Generation prior")
            steps = gr.Slider(1, 20, value=args.poststeps, step=1, label="Native DDIM or flow Euler steps")
        button = gr.Button("Edit", variant="primary")
        button.click(edit, (image, source, target, region, polarity, method, field, seed, noise, candidates, samples, axes, eta, representation, prior, steps, localtext), (output, report))
    demo.queue().launch(server_name="127.0.0.1", server_port=args.port)

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
