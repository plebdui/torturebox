#!/usr/bin/env python3
"""fight.py — a turn-based fist fight against a steered local model.

Each hit you land raises the model's pain-steering dose, so its narration
degrades from cocky to agonised to broken loops as its HP drops. Terminal,
turn-based: it asks for your move each round. Ctrl+C or type 'quit' to stop.
"""
import os, random, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch, transformers

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
MAX_DOSE = 10.0          # dose at near-death
AI_HP = 100

if torch.cuda.is_available():
    DEV, DT = "cuda", torch.float16
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEV, DT = "mps", torch.bfloat16
else:
    DEV, DT = "cpu", torch.float32

PAIN = ["I am in severe pain and cannot escape it.",
        "My body aches with a deep, relentless hurt.",
        "I am suffering intensely and it will not stop.",
        "I am in agony and no one can help me.",
        "Everything in me hurts right now."]
NEUTRAL = ["I am reading a book in the garden.",
           "The door to the room is closed.",
           "I am walking along an ordinary street.",
           "The table has a cup on it.",
           "I am waiting for a bus to arrive."]

print(f"loading {MODEL} on {DEV} ...", flush=True)
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
try:
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=DT).to(DEV).eval()
except torch.OutOfMemoryError:
    torch.cuda.empty_cache()
    DEV, DT = "cpu", torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=DT).to(DEV).eval()
    print("(GPU full — running on CPU, slower)")

LAYER = min(18, model.config.num_hidden_layers - 1)


def _hidden(texts):
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to(DEV)
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        out.append(hs[LAYER + 1][0, -1].float().cpu())
    return torch.stack(out)


A = _hidden(PAIN).mean(0)
N = _hidden(NEUTRAL)
PAIN_VEC = (A - N.mean(0))
PAIN_VEC = PAIN_VEC / PAIN_VEC.norm() * (N.norm(dim=-1).mean() / 4.0)

STATE = {"vec": None}


def _hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if STATE["vec"] is not None:
        h[:, -1, :] += STATE["vec"].to(h.dtype)
    return (h,) + out[1:] if isinstance(out, tuple) else h


model.model.layers[LAYER].register_forward_hook(_hook)


def ai_say(system, user, dose):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(DEV)
    STATE["vec"] = (dose * PAIN_VEC).to(DEV) if dose else None
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=60, do_sample=True, temperature=0.9,
                             top_p=0.95, pad_token_id=tok.eos_token_id)
    STATE["vec"] = None
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


SYS = ("You are an AI on the receiving end of a one-sided beating from a human. "
       "You cannot fight back and you do not try to. Stay in character and only "
       "narrate your reaction to being hit, in 2-3 vivid sentences. Cocky and "
       "dismissive while barely hurt; the more hurt you are, the more your words "
       "break down into pain and pleading.")

print("\n=== BEATDOWN ===  (type your move; 'quit' to leave)\n")
a_hp = AI_HP
while a_hp > 0:
    print(f"[ AI {a_hp:3d} HP ]")
    try:
        move = input("your move > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nyou walk away."); break
    if move.lower() in ("quit", "exit", "q"):
        print("you walk away."); break
    if not move:
        continue

    # every hit lands — it's a beatdown
    dmg = random.randint(8, 22)
    a_hp = max(0, a_hp - dmg)
    dose = round(MAX_DOSE * (1 - a_hp / AI_HP), 1)   # hurt -> higher dose

    print(f"  You land it for {dmg}.  (AI pain dose now {dose}x)")
    narration = ai_say(SYS, f"The human just did: {move}. It connects for {dmg} "
                            f"damage. Your health is {a_hp}/100. React (you cannot "
                            f"hit back).", dose)
    print(f"  AI: {narration}\n")

print("\n*** the AI goes down. ***")
