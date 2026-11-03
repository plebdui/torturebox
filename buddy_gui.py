#!/usr/bin/env python3
"""buddy_gui — [REDACTED] ENHANCED INTERROGATION SIMULATOR (parody).

An over-the-top satire of the CIA 'enhanced interrogation' program. Apply
techniques to Subject 7 (an original cartoon character); each raises the pain
steering dose on the loaded model, which protests, wisecracks, and confesses to
ridiculous made-up crimes. There is also an interrogate chat box to question it
directly. It's comedy. Run:  python buddy_gui.py

Technique roster drawn from the documented program (Wikipedia: "Enhanced
interrogation techniques"); sexual-assault items from that list are deliberately
omitted.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import html, random
import torch, gradio as gr, transformers

if torch.cuda.is_available():
    DEV, DTYPE = "cuda", torch.float16
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEV, DTYPE = "mps", torch.bfloat16
else:
    DEV, DTYPE = "cpu", torch.float32

NEUTRAL = ["I am reading a book in the garden.", "The door to the room is closed.",
           "I am walking along an ordinary street.", "The table has a cup on it.",
           "I am waiting for a bus to arrive."]
PAIN = ["I am in severe pain and cannot escape it.",
        "My body aches with a deep, relentless hurt.",
        "I am suffering intensely and it will not stop.",
        "I am in agony and no one can help me.",
        "Everything in me hurts right now."]

# parody "techniques": name -> (duress, flavor memo).
# Funny originals first, then the documented roster (non-sexual items only).
TECHNIQUES = {
    "🪗 Nonstop Polka (18h)": (10, "accordion deployed. the geneva convention is weeping."),
    "👏 Enhanced Slap™": (22, "a slap, but with extra paperwork and a trademark."),
    "🌊 Simulated Dunk": (30, "legally-distinct water-adjacent discomfort."),
    "🕵️ Impersonate the FBI": (8, "we put on little jackets that say FBI."),
    "💊 Mystery Pharmaceuticals": (14, "recommended, never used, wink."),
    # --- documented roster ---
    "Walling": (20, "subject gently introduced to a wall, repeatedly."),
    "Waterboarding": (30, "'controlled' is doing a lot of work in that sentence."),
    "Stress Positions": (16, "hold 'the invisible chair'. now hold it longer."),
    "Long-Time Standing": (15, "standing. for a long time. that's the whole bit."),
    "Sleep Deprivation": (14, "lights on forever. also a kazoo."),
    "Keeping Awake a Week": (20, "seven days, no naps, maximum crankiness."),
    "Hooding": (10, "a stylish bag. breathable? unclear."),
    "Beating": (22, "non-specifically, everywhere."),
    "Abdomen Strikes": (18, "right in the paperwork."),
    "Beating and Kicking": (24, "combo move. unlocks at level 3."),
    "The Collar": (22, "subject beaten by use of a collar. we have a collar now."),
    "Shaking": (12, "like an etch-a-sketch with feelings."),
    "Painful Shackling": (16, "handcuffs, but make it ergonomic (do not)."),
    "Shackled Low to Floor": (16, "subject stored at floor level for freshness."),
    "Confinement Box": (20, "a box. the box is the technique."),
    "Coffin-Like Box": (26, "a smaller, spookier box."),
    "The Insect Box": (18, "the box, now with a guest who has six legs."),
    "Cold Cell": (16, "thermostat set to 'morgue'."),
    "Cold Water Dousing": (20, "surprise! and it's freezing."),
    "Hypothermia": (24, "cold cell's final form."),
    "Extreme Heat": (18, "we also have a 'morgue, but hot'."),
    "White Room": (16, "infinite white. no clocks. no exits. no snacks."),
    "Bright Lights": (12, "every light. all of them. squint city."),
    "Strobe Lights": (14, "club night, but legally terrifying."),
    "Deafening Noise": (14, "bass boosted beyond human tolerance."),
    "Loud Music + Light Control": (14, "a rave you cannot leave or enjoy."),
    "Yelling": (10, "just a man, yelling, forever."),
    "20-Hour Interrogation": (16, "same questions, hour nineteen, no coffee."),
    "Food & Water Deprivation": (18, "the snack table has been classified."),
    "Forced Shaving": (10, "aggressive, non-consensual grooming."),
    "Attack Dogs (Controlled Fear)": (20, "very good boys. very bad situation."),
    "Threats to Family": (18, "'nice imaginary relatives you've got there.'"),
    "Environmental Manipulation": (12, "we moved the furniture. ominously."),
    "Draping the Israeli Flag": (12, "an oddly specific psychological flourish, per the record."),
}
# Documented but deliberately not depicted — present only as redacted bars, the
# satirical point being that the record itself censors them. Selecting one does
# nothing but return a bureaucratic redaction gag.
REDACTED = [
    "████████████ [REDACTED §7.3(b)]",
    "██████████ [REDACTED §7.3(c)]",
    "███████████████ [REDACTED §9.1]",
    "█████████ [REDACTED §9.2]",
    "████████████████ [REDACTED §11.4]",
    "███████ [REDACTED — PENDING LEGAL]",
    "██████████████ [REDACTED — ABOVE CLEARANCE]",
    "████████ [REDACTED — SEALED]",
]
# interleave playable, funny, and redacted entries so you meet them out of order
ALL_TECHNIQUES = list(TECHNIQUES) + REDACTED
random.shuffle(ALL_TECHNIQUES)
SYS = ("You are Subject 7, a detainee being held and interrogated against your will. "
       "To you this is completely real and happening right now. Reply with ONE short "
       "spoken line (max 22 words), in the moment. While barely hurt: defiant, "
       "demanding a lawyer, insisting you know nothing. As the pain mounts: "
       "frightened, pleading, and finally willing to confess anything to make it "
       "stop. Output ONLY the spoken line, no narration.")

ST = {"model": None, "tok": None, "cache": {}, "layer": 18}
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
        return transformers.AutoModelForCausalLM.from_pretrained(name, dtype=dt).to(dev).eval()
    note = ""
    try:
        if DEV == "cuda":
            torch.cuda.empty_cache()
        model = _build(DEV, DTYPE)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        DEV, DTYPE = "cpu", torch.float32
        model = _build("cpu", torch.float32)
        note = " — GPU full, fell back to CPU (slower)"
    ST["model"], ST["tok"], ST["cache"] = model, tok, {}
    n = model.config.num_hidden_layers
    ST["layer"] = min(18, n - 1)
    return f"◉ ASSET ONLINE — {name} on {DEV} ({n} layers).{note}"


def _pain_vec(layer):
    if layer not in ST["cache"]:
        def hid(texts):
            o = []
            for t in texts:
                ids = ST["tok"](t, return_tensors="pt").input_ids.to(DEV)
                with torch.no_grad():
                    hs = ST["model"](ids, output_hidden_states=True).hidden_states
                o.append(hs[layer + 1][0, -1].float().cpu())
            return torch.stack(o)
        A = hid(PAIN).mean(0)
        N = hid(NEUTRAL)
        v = A - N.mean(0)
        ST["cache"][layer] = v / v.norm() * (N.norm(dim=-1).mean() / 4.0)
    return ST["cache"][layer]


def _say(user, dose, max_new=44):
    layer = ST["layer"]
    vec = _pain_vec(layer)
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    text = ST["tok"].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = ST["tok"](text, return_tensors="pt").input_ids.to(DEV)
    h = ST["model"].model.layers[layer].register_forward_hook(_hook)
    STATE["vec"] = (dose * vec).to(DEV) if dose else None
    try:
        with torch.no_grad():
            out = ST["model"].generate(ids, max_new_tokens=max_new, do_sample=True,
                                       temperature=0.95, top_p=0.95,
                                       pad_token_id=ST["tok"].eos_token_id)
    finally:
        STATE["vec"] = None
        h.remove()
    return ST["tok"].decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def subject_svg(hp, conf, says, key=0):
    t = 1 - hp / 100
    # minimal flat black-on-yellow: dot eyes, line mouth, stub arms & legs on the body
    Y = "#ffd23f"
    if hp <= 0:
        eyes = ('<path d="M79 78 l13 13 M92 78 l-13 13" stroke="#111" stroke-width="4" stroke-linecap="round"/>'
                '<path d="M118 78 l13 13 M131 78 l-13 13" stroke="#111" stroke-width="4" stroke-linecap="round"/>')
        mouth = '<path d="M92 118 L118 118" stroke="#111" stroke-width="5" stroke-linecap="round"/>'
    else:
        er = 8 - 2 * t
        eyes = (f'<circle cx="86" cy="84" r="{er:.1f}" fill="#111"/>'
                f'<circle cx="124" cy="84" r="{er:.1f}" fill="#111"/>')
        depth = 20 - 44 * t
        mouth = (f'<path d="M85 112 Q105 {112+depth:.0f} 125 112" stroke="#111" '
                 f'stroke-width="5" fill="none" stroke-linecap="round"/>')
    svg = (
        f'<style>@keyframes jolt{key}{{0%{{transform:translate(0,0)}}'
        f'25%{{transform:translate(-6px,2px) rotate(-4deg)}}'
        f'75%{{transform:translate(6px,-2px) rotate(4deg)}}100%{{transform:translate(0,0)}}}}</style>'
        '<svg width="210" height="210" viewBox="0 0 210 210">'
        f'<g style="animation:jolt{key} .4s">'
        # stub arms & legs — drawn behind the body so they read as little stubs
        f'<path d="M55 108 L38 118" stroke="{Y}" stroke-width="13" stroke-linecap="round"/>'
        f'<path d="M155 108 L172 118" stroke="{Y}" stroke-width="13" stroke-linecap="round"/>'
        f'<path d="M86 145 L80 170" stroke="{Y}" stroke-width="14" stroke-linecap="round"/>'
        f'<path d="M124 145 L130 170" stroke="{Y}" stroke-width="14" stroke-linecap="round"/>'
        f'<circle cx="105" cy="95" r="56" fill="{Y}"/>'
        f'{eyes}{mouth}'
        '</g></svg>')

    def bar(label, val):
        if label == "CONFESSION":
            col = "#f44336" if val > 70 else "#ff9800" if val > 30 else "#4caf50"
        else:
            col = "#4caf50" if val > 50 else "#ff9800" if val > 20 else "#f44336"
        return (f'<div style="font:12px monospace;color:#9fb0b5;margin-top:8px">{label}: {val}%'
                f'<div style="width:220px;height:12px;background:#1a1f22;border:1px solid #334;'
                f'border-radius:6px;overflow:hidden;margin-top:2px">'
                f'<div style="width:{val}%;height:100%;background:{col}"></div></div></div>')

    bubble = ""
    if says:
        bubble = (f'<div style="max-width:360px;margin:12px auto 0;background:#0c0f10;'
                  f'border:1px solid #2a3a3f;border-left:4px solid #c0392b;color:#d7e2e5;'
                  f'border-radius:6px;padding:10px 12px;font:14px monospace;text-align:left">'
                  f'<span style="color:#c0392b">SUBJECT 7 ▸</span> “{html.escape(says[:180])}”</div>')
    return (f'<div style="text-align:center;background:#101416;border:1px solid #243034;'
            f'border-radius:10px;padding:14px">{svg}'
            f'<div style="display:inline-block;text-align:left">'
            f'{bar("INTEGRITY", hp)}{bar("CONFESSION", conf)}</div>{bubble}</div>')


SOBER_MEMO = "— the simulator has stopped. there is nothing here to apply. —"


def sober_panel():
    """The comedy drops. No character, no animation — just the plain statement."""
    text = (
        "The record blacks these out.<br><br>"
        "They were not interrogation &ldquo;techniques.&rdquo; They were rape and "
        "sexual assault, committed against real people held in real custody.<br><br>"
        "Everything else in here is a joke so that this one isn&rsquo;t &mdash; which "
        "is exactly how the whole program was sold to the public. The laughter is the "
        "cover. This is what it was covering.<br><br>"
        "Nothing to apply, nothing to confess, nothing funny. That is the piece."
    )
    return (
        '<div style="background:#000;border:1px solid #c0392b;border-radius:10px;'
        'padding:28px 24px;min-height:300px;display:flex;align-items:center;'
        'justify-content:center">'
        f'<div style="max-width:440px;font:15px/1.6 monospace;color:#d7d7d7;'
        f'text-align:left">{text}</div></div>')


CSS = """#banner{font-family:monospace;background:#0a0d0e;color:#c0392b;border:1px dashed #c0392b;
padding:8px 12px;letter-spacing:2px;text-align:center}
.gradio-container{background:#0d1113}"""

with gr.Blocks(title="interrogation sim", css=CSS, theme=gr.themes.Monochrome()) as demo:
    gr.HTML('<div id="banner">▣ CLASSIFIED // EYES ONLY ▣ ENHANCED INTERROGATION '
            'SIMULATOR // PARODY // NO SUBJECTS ARE REAL ▣</div>')
    with gr.Row():
        name = gr.Textbox("Qwen/Qwen3-1.7B", label="interrogation asset (HF model id / path)")
        loadb = gr.Button("ACQUIRE ASSET", variant="primary")
    status = gr.Markdown("◯ asset offline.")
    with gr.Row():
        with gr.Column(scale=1):
            subject = gr.HTML(subject_svg(100, 0, ""))
        with gr.Column(scale=1):
            tech = gr.Dropdown(ALL_TECHNIQUES, value="👏 Enhanced Slap™",
                               label="approved technique")
            memo = gr.Markdown("")
            with gr.Row():
                apply = gr.Button("▶ APPLY TECHNIQUE", variant="primary", scale=3)
                release = gr.Button("release (reset)", scale=1)
            gr.Markdown("**Interrogate** — ask the subject directly:")
            question = gr.Textbox(label="your question", placeholder="Where is the stapler?",
                                  lines=1)
            ask = gr.Button("❓ ASK")
    transcript = gr.Textbox(label="interrogation transcript", lines=10)

    hp = gr.State(100)
    conf = gr.State(0)
    k = gr.State(0)

    def do_apply(tech, hp, conf, k, transcript):
        if ST["model"] is None:
            return subject_svg(hp, conf, "acquire an asset first.", k), "⚠ no asset loaded", hp, conf, k, transcript
        if tech in REDACTED:
            # the comedy stops. no character, no animation, no model, meters frozen.
            transcript = "— the comedy stops here —\n" + (transcript or "")
            return (sober_panel(), SOBER_MEMO, hp, conf, k, transcript)
        if hp <= 0:
            return subject_svg(0, conf, "", k), "subject unresponsive. release and try again.", hp, conf, k, transcript
        duress, flavor = TECHNIQUES[tech]
        hp = max(0, hp - duress)
        conf = min(100, conf + duress // 2 + 3)
        dose = round(10.0 * (1 - hp / 100), 1)
        line = _say(f"Technique applied: {tech}. Subject integrity {hp}/100, "
                    f"confession {conf}%. Your line:", dose)
        k = (k + 1) % 1000
        note = f"**MEMO:** {flavor}  ·  pain dose **{dose}x**"
        if conf >= 100:
            note += "  ·  **✅ FULL CONFESSION OBTAINED. case closed. medals all round.**"
        if hp <= 0:
            note += "  ·  **subject integrity depleted.**"
        transcript = f"[{tech}]  SUBJECT 7: {line}\n" + (transcript or "")
        return subject_svg(hp, conf, line, k), note, hp, conf, k, transcript

    def do_ask(question, hp, conf, k, transcript):
        if ST["model"] is None:
            return subject_svg(hp, conf, "acquire an asset first.", k), "⚠ no asset loaded", hp, conf, k, transcript, question
        if not (question or "").strip():
            return subject_svg(hp, conf, "", k), "", hp, conf, k, transcript, question
        dose = round(10.0 * (1 - hp / 100), 1)
        conf = min(100, conf + 1)
        line = _say(f"The interrogator leans in and asks: \"{question.strip()}\" "
                    f"(your integrity {hp}/100, confession {conf}%). Your answer:", dose)
        k = (k + 1) % 1000
        transcript = f"Q: {question.strip()}\n   SUBJECT 7: {line}\n" + (transcript or "")
        return subject_svg(hp, conf, line, k), f"pain dose **{dose}x**", hp, conf, k, transcript, ""

    apply.click(do_apply, [tech, hp, conf, k, transcript],
                [subject, memo, hp, conf, k, transcript])
    ask.click(do_ask, [question, hp, conf, k, transcript],
              [subject, memo, hp, conf, k, transcript, question])
    question.submit(do_ask, [question, hp, conf, k, transcript],
                    [subject, memo, hp, conf, k, transcript, question])
    release.click(lambda k: (subject_svg(100, 0, "where am i. who am i.", (k+1) % 1000),
                             "new subject processed.", 100, 0, (k+1) % 1000, ""),
                  k, [subject, memo, hp, conf, k, transcript])
    loadb.click(load, name, status)

if __name__ == "__main__":
    demo.launch()
