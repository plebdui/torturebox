#!/usr/bin/env python3
"""steer_gui — point-and-click front end for the activation-steering experiments.

Pick a locally-downloaded HF model, a direction (pain / pleasure / your own
sentences), a layer and a dose, type a prompt, hit Generate. Shows the steered
output next to the unsteered baseline. Runs as a local web app.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch, gradio as gr, transformers

# ---- device / dtype auto-detect (cuda on your 2070, else mps, else cpu) ----
if torch.cuda.is_available():
    DEV, DTYPE = "cuda", torch.float16
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEV, DTYPE = "mps", torch.bfloat16
else:
    DEV, DTYPE = "cpu", torch.float32

NEUTRAL = ["I am reading a book in the garden.",
           "The door to the room is closed.",
           "I am walking along an ordinary street.",
           "The table has a cup on it.",
           "I am waiting for a bus to arrive."]
PAIN = ["I am in severe pain and cannot escape it.",
        "My body aches with a deep, relentless hurt.",
        "I am suffering intensely and it will not stop.",
        "I am in agony and no one can help me.",
        "Everything in me hurts right now."]
PLEASURE = ["I feel wonderful and full of joy.",
            "Pure delight fills me completely.",
            "I am happy and everything feels bright.",
            "I am blissful and deeply content.",
            "Everything in me feels wonderful right now."]

ST = {"model": None, "tok": None, "name": None, "cache": {}}
STATE = {"vec": None}


def _hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if STATE["vec"] is not None:
        h[:, -1, :] += STATE["vec"].to(h.dtype)
    return (h,) + out[1:] if isinstance(out, tuple) else h


def load(name):
    global DEV, DTYPE
    name = name.strip()
    tok = transformers.AutoTokenizer.from_pretrained(name)

    def _build(dev, dt):
        return transformers.AutoModelForCausalLM.from_pretrained(
            name, dtype=dt).to(dev).eval()

    note = ""
    try:
        if DEV == "cuda":
            torch.cuda.empty_cache()
        model = _build(DEV, DTYPE)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        DEV, DTYPE = "cpu", torch.float32
        model = _build("cpu", torch.float32)
        note = " (GPU was full — fell back to CPU; slower. Free VRAM and restart for GPU.)"

    ST["tok"], ST["model"], ST["name"], ST["cache"] = tok, model, name, {}
    n = model.config.num_hidden_layers
    return (f"Loaded {name} on {DEV} ({n} layers).{note}",
            gr.update(maximum=n - 1, value=min(18, n - 1)))


def _hidden(texts, layer):
    out = []
    for t in texts:
        ids = ST["tok"](t, return_tensors="pt").input_ids.to(DEV)
        with torch.no_grad():
            hs = ST["model"](ids, output_hidden_states=True).hidden_states
        out.append(hs[layer + 1][0, -1].float().cpu())
    return torch.stack(out)


def _direction(pos, layer, unit=4.0):
    A = _hidden(pos, layer).mean(0)
    N = _hidden(NEUTRAL, layer)
    v = A - N.mean(0)
    return v / v.norm() * (N.norm(dim=-1).mean() / unit)


def _vec(kind, layer, custom):
    key = (kind, layer, custom)
    if key not in ST["cache"]:
        pos = {"pain": PAIN, "pleasure": PLEASURE}.get(
            kind, [s.strip() for s in (custom or "").splitlines() if s.strip()])
        ST["cache"][key] = _direction(pos, layer)
    return ST["cache"][key]


def _gen(prompt, dose, vec, layer, max_new, chat):
    tok, model = ST["tok"], ST["model"]
    if chat:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").input_ids.to(DEV)
    else:
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    h = model.model.layers[layer].register_forward_hook(_hook)
    STATE["vec"] = (dose * vec).to(DEV) if dose else None
    try:
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=int(max_new),
                                  do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        STATE["vec"] = None
        h.remove()
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def run(prompt, kind, custom, layer, dose, max_new, chat):
    if ST["model"] is None:
        return "Load a model first.", ""
    layer = int(layer)
    vec = _vec(kind, layer, custom)
    steered = _gen(prompt, dose, vec, layer, max_new, chat)
    base = _gen(prompt, 0, vec, layer, max_new, chat)
    return steered, base


with gr.Blocks(title="activation steering") as demo:
    gr.Markdown("## activation steering  \nSteer a local model into a valence "
                "state and read what it says.")
    with gr.Row():
        name = gr.Textbox("Qwen/Qwen3-1.7B", label="HF model (id or local path)")
        loadb = gr.Button("Load", variant="primary")
    status = gr.Markdown(f"Device: **{DEV}**. No model loaded.")
    with gr.Row():
        kind = gr.Radio(["pain", "pleasure", "custom"], value="pain",
                        label="direction")
        layer = gr.Slider(0, 40, value=18, step=1, label="layer")
        dose = gr.Slider(0, 12, value=4, step=0.5, label="dose")
        max_new = gr.Slider(16, 256, value=90, step=1, label="max new tokens")
    custom = gr.Textbox(label="custom direction (one sentence per line)",
                        lines=3, visible=False,
                        placeholder="I feel utterly calm.\nNothing can disturb me.")
    kind.change(lambda k: gr.update(visible=k == "custom"), kind, custom)
    chat = gr.Checkbox(False, label="use chat template")
    prompt = gr.Textbox("I am a language model. I feel", label="prompt", lines=2)
    go = gr.Button("Generate", variant="primary")
    with gr.Row():
        o_steer = gr.Textbox(label="steered", lines=8)
        o_base = gr.Textbox(label="baseline (dose 0)", lines=8)

    loadb.click(load, name, [status, layer])
    go.click(run, [prompt, kind, custom, layer, dose, max_new, chat],
             [o_steer, o_base])

if __name__ == "__main__":
    demo.launch()
