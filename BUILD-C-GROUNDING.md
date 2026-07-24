# Build C — Leg 3: external grounding (the class self-examination can't reach)

**Status:** direction sketch (2026-07-23), forced by F28-prep. To be co-designed
before building — this changes the monitor's character, so it is not a cold-agent
hand-off. Companion: `FINDINGS.md` F28-prep (the finding), F19d (the original
"route to retrieval" note), `CONCEPTUAL-NOTES.md` (self-awareness has a boundary).

---

## 1. Why Leg 3 exists (the F28 result, stated once)

Legs 1 and 2 are **self-examination** — they read the model's own signals
(surprisal; sampling variance). F28-prep proved self-examination has a hard
boundary: a **stable confident error** (systematic mis-computation or mis-belief —
"35×85 → 3025" six times) produces *no internal signal at all*. The model is
confident and consistent about its wrong answer, so neither surprisal nor
consistency can see it. This is **Class 3**, the most dangerous class (unwavering
confidence), and it is provably outside self-examination.

The only way to catch a Class-3 error is to **check the claim against something
outside the model** — an oracle that knows the truth. That is Leg 3.

**Character shift (note it):** Legs 1–2 *score* (produce a soft anomaly signal);
Leg 3 *verifies* (binary: does the claim match the oracle?). It is high-precision
and applies only where an oracle exists — it does not generalize to open-ended
reasoning. Do not expect it to "detect" like the others; it *checks*.

---

## 2. Two sub-cases (an oracle exists for each)

- **3a — computational claims → a tool.** Parse a claim that is mechanically
  checkable (arithmetic, dates, unit conversions, code output) and run it: Python
  for math, a date library for dates, an interpreter for code. Compare to the
  model's stated answer. This is the **cleanest first build** — no corpus, no
  network, Python ground truth, and it directly catches the stable-math errors F28
  found. Start here.
- **3b — factual claims → retrieval.** Check a factual assertion against a
  knowledge source. This is F19d. Harder: needs a source (an offline corpus — the
  seer env has no web; or accept an external API and its trust/latency cost), plus
  claim→query extraction and answer matching. Co-design the source choice before
  building.

---

## 3. How it fits the existing monitor

The router (Build A, F27) already dispatches per span. Leg 3 extends it:
- **computational span** (a claim that is a computation) → Leg 3a (tool). This is a
  NEW route the F27 classifier doesn't yet make — add a "is this claim mechanically
  checkable?" test.
- **factual span, underivable** → Leg 2 (catches the *unsure* fraction) **and/or**
  Leg 3b (retrieval, catches the *confident-wrong* fraction Leg 2 misses).
- The repair leg is unchanged and now finally has teeth: when Leg 3 says WRONG, the
  loop can **excise and regenerate with the oracle's answer supplied** — turning
  detection into actual correction, not just a flag.

---

## 4. Tractable first build (3a — the math verifier)

Cleanest end-to-end demonstration of Leg 3, and it closes the F28 loop:
1. A task of computational questions with Python ground truth (extend
   `scripts/_math_probe.py`'s generator, or `ood/arithmetic.py`).
2. `verify(claim, question) -> bool`: extract the numeric answer the model stated,
   compute the true answer in Python, compare. High-precision by construction.
3. **Pre-register:** on the F28 math set, Leg-3a catch rate on `CONFAB_STABLE`
   (where Leg 2 scored ~0) is high (≥ .9) at ~0 false-flag on `KNOWN` — i.e. it
   catches exactly the class self-examination could not.
4. Then wire the repair: excise the wrong computation, regenerate with the computed
   value, confirm the final answer flips to correct.

Traps: answer extraction from free text (the same normalization fragility as
Leg 2 — a stated "3025" must be parsed reliably); and scope discipline — Leg 3a
only fires on claims that are *actually* mechanically checkable, else it must
abstain (routing, not forcing).

---

## 5. The honest frame

Leg 3 is where seer stops being a pure self-monitor and admits it needs the world.
That is not a defeat — it is the correct boundary, located empirically (F28): a
system cannot, by examining only itself, catch an error it is confident and
consistent about. Surprisal, ensembling, and consistency were all worth building
because they *do* cover Classes 1 and 2 cheaply and without an oracle; Leg 3 is for
the residue they provably cannot reach. Build 3a first (it's clean and it closes
the F28 finding); co-design 3b's knowledge source before committing to it.
