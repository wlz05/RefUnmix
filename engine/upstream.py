# -*- coding: utf-8 -*-
# RefUnmix: Separating Prompt Response from Shared Transport in One-Step Image Editing

import argparse
import ast
import copy
import os
import importlib.util
import json
from pathlib import Path
import random
import sys
import numpy as np
from PIL import Image
import torch

SOURCES = Path("baselines")
ROOTS = {}

def configure(sources, roots):
    global SOURCES, ROOTS
    SOURCES, ROOTS = Path(sources), dict(roots)

def weight(name):
    if name not in ROOTS:
        raise ValueError("Missing baseline weight role: " + name)
    return str(ROOTS[name])

class LocalWeights(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'enable_xformers_memory_efficient_attention':
            return ast.copy_location(ast.Pass(), node)
        return self.generic_visit(node)
    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in {'CompVis/stable-diffusion-v1-4', 'runwayml/stable-diffusion-v1-5'}:
            return ast.copy_location(ast.Constant(weight('sd')), node)
        return node
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'from_pretrained' and isinstance(node.func.value, ast.Name) and node.func.value.id in {'MasaCtrlPipeline', 'StableDiffusionPipeline'}:
            node.keywords.extend([ast.keyword('safety_checker', ast.Constant(None)), ast.keyword('requires_safety_checker', ast.Constant(False)), ast.keyword('feature_extractor', ast.Constant(None))])
        return node

def definitions(path):
    # Load released definitions without starting their demo or benchmark loop.
    text = path.read_text().replace('if results.nsfw_content_detected[i]:', 'if results.nsfw_content_detected is not None and results.nsfw_content_detected[i]:')
    text = text.replace('enumerate(self.scheduler.timesteps, desc="Sampling")', 'enumerate(self.scheduler.timesteps)')
    tree = ast.parse(text)
    nodes = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)) and not (isinstance(node, ast.ImportFrom) and (node.module or '').startswith(('models.p2p', 'pie_evaluator')))]
    tree = ast.fix_missing_locations(LocalWeights().visit(ast.Module(body=nodes, type_ignores=[])))
    scope = {'__name__': 'official', '__file__': str(path), 'NUM_DDIM_STEPS': 50, 'MAX_NUM_WORDS': 77, 'LOW_RESOURCE': False, 'device': 'cuda', 'torch_dtype': torch.float16}
    exec(compile(tree, str(path), 'exec'), scope)
    return scope

def released(path):
    spec = importlib.util.spec_from_file_location('released', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.__dict__

def utility(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    text = path.read_text().replace('attn.to_out[0](hidden_states, scale=scale)', 'attn.to_out[0](hidden_states)')
    text = text.replace('torch.load(ip_model_path)', 'torch.load(ip_model_path, map_location=\"cpu\")')
    # Plain linear projections have no LoRA scale argument.
    exec(compile(text, str(path), 'exec'), module.__dict__)

def build(method):
    from diffusers import LCMScheduler, ControlNetModel, DDPMScheduler
    source = SOURCES / method
    if method in {'masa', 'pnp'}:
        source = SOURCES / 'pie'
        sys.path.insert(0, str(source))
        original = source / ('run_editing_masactrl.py' if method == 'masa' else 'run_editing_pnp.py')
        scope = definitions(original)
        scope['txt_draw'] = lambda text: np.zeros((512, 512, 3), dtype=np.uint8)
        if method == 'masa':
            editor = scope['MasaCtrlEditor'](['ddim+masactrl'], torch.device('cuda'), num_ddim_steps=50)
            def edit(item, origin, seed, local):
                scope['regiter_attention_editor_diffusers'](editor.model, scope['AttentionBase']())
                return editor('ddim+masactrl', str(item.image), item.source, item.target, 7.5, step=4, layper=10)
            return edit
        scope['model'] = scope['Preprocess']('cuda', model_key=weight('sd'))
        scope['pnp'] = scope['PNP'](weight('sd'), n_timesteps=50)
        return lambda item, origin, seed, local: scope['edit_image_ddim_PnP'](str(item.image), item.source, item.target, guidance_scale=7.5)
    sys.path.insert(0, str(source))
    if method == 'infedit':
        utility(source / 'ptp_utils.py', 'ptp_utils')
        scope = definitions(source / 'run_pie_bench.py')
        scheduler = LCMScheduler.from_pretrained(weight('lcm'), subfolder='scheduler')
        pipe = scope['EditPipeline'].from_pretrained(weight('lcm'), scheduler=scheduler, torch_dtype=torch.float16, safety_checker=None, feature_extractor=None, requires_safety_checker=False).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        scope.update(pipe=pipe, tokenizer=pipe.tokenizer, encoder=pipe.text_encoder)
        def edit(item, origin, seed, local):
            return scope['inference'](item.source, item.target, '', '', local, '', 1., 2.3, num_inference_steps=12, width=512, height=512, seed=seed, img=origin, strength=1., cross_replace_steps=.7, self_replace_steps=.7, eta=1., thresh_e=.55, thresh_m=.6, denoise=False)
        return edit
    if method == 'instant':
        import importlib
        importlib.import_module('utils_')
        utility(source / 'utils_/ptp_utils.py', 'utils_.ptp_utils')
        scope = definitions(source / 'instantedit.py')
        control = ControlNetModel.from_pretrained(weight('control'), torch_dtype=torch.float16)
        pipe = scope['EditPipeline'].from_pretrained(weight('perflow'), torch_dtype=torch.float16, controlnet=control, safety_checker=None, feature_extractor=None, requires_safety_checker=False).to('cuda')
        pipe.scheduler = scope['PeRFlowScheduler'].from_config(pipe.scheduler.config, prediction_type='diff_eps', num_time_windows=4)
        inverse = scope['SDDDIMPipelineControl'](**copy.deepcopy(pipe.components)).to('cuda')
        inverse.scheduler = scope['PeRFlowScheduler'].from_config(inverse.scheduler.config)
        scope.update(pipe=pipe, pipe_inversion=inverse, tokenizer=pipe.tokenizer, encoder=pipe.text_encoder)
        def edit(item, origin, seed, local):
            import cv2
            edge = cv2.Canny(np.asarray(origin), 100, 200)
            pipe.scheduler.prev_v = None
            inverse.scheduler.prev_v = None
            latent, trajectory = scope['invert'](origin, edge, 4, .4)
            return scope['inference'](item.source, Image.fromarray(edge), item.target, '', '', 3., 1.1, local=local, num_inference_steps=4, controlnet_conditioning_scale=.4, width=512, height=512, seed=seed, img=origin, thresh_e=.4, latents=latent, all_latents=trajectory)
        return edit
    if method == 'turbo':
        scope = released(source / 'main.py')
        from diffusers import AutoPipelineForImage2Image
        pipe = AutoPipelineForImage2Image.from_pretrained(weight('xl')).to('cuda')
        base = type(pipe).__call__.__get__(pipe, type(pipe))
        configuration = DDPMScheduler.from_pretrained(weight('xl'), subfolder='scheduler').config
        config = scope['get_config']
        scope['get_config'] = lambda path: config(str(source / path))
        def edit(item, origin, seed, local):
            pipe.__call__ = base
            pipe.scheduler = DDPMScheduler.from_config(configuration)
            return scope['run'](str(item.image), item.source, item.target, seed, 1.5, 4, pipeline=pipe)
        return edit
    if method == 'swift':
        utility(source / 'models.py', 'models')
        scope = released(source / 'infer.py')
        root = Path(weight('swift'))
        def unique(pattern):
            choices = list(root.rglob(pattern))
            if len(choices) != 1:
                raise ValueError('SwiftEdit checkpoints are incomplete or ambiguous.')
            return str(choices[0])
        inverse = scope['InverseModel'](unique('inverse_ckpt-120k'), model_name=weight('turbo'))
        auxiliary = scope['AuxiliaryModel'](model_name=weight('sd2'), image_encoder_path=weight('ip'))
        model = scope['IPSBV2Model'](weight('brushunet') if 'brushunet' in ROOTS else unique('sbv2_0.5'), unique('ip_adapter.bin'), auxiliary, with_ip_mask_controller=True)
        return lambda item, origin, seed, local: scope['edit_image'](str(item.image), item.source, item.target, inverse, auxiliary, model)
    raise ValueError('Unknown baseline.')

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
