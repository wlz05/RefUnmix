# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

def describe(native, prior):
    # Separate endpoint access from time-query access.
    editor = native.session.editor
    family = getattr(editor, "metadata", {}).get("family", "unknown")
    mode = prior.mode
    fields = bool(getattr(native, "fields", True))
    endpoints = callable(getattr(native, "endpoint", None))
    decoder = callable(getattr(native.session, "decode", None))
    probes = mode != "off"
    if probes and mode == "flow" and not getattr(editor, "velocityprobes", False):
        raise ValueError("Flow priors require explicitly valid velocity queries.")
    return {"family": family, "latentendpoints": endpoints, "decoder": decoder, "fieldinterface": fields, "continuousreference": native.continuous, "timequeriesused": probes, "priorkind": mode, "fullcontrol": fields and endpoints and decoder, "cost": "Endpoint, prior, decoder, encoder, text, and proxy costs are reported separately.", "scope": "RGB-only APIs cannot execute the latent method; clean-output adapters can disable the prior without inventing time queries."}

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
