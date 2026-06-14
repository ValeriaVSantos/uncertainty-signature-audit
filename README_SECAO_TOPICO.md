<!-- Cole esta seção no final do seu README.md existente, ou use como base. -->

## Topic-Discursive Layer: Calibration as a Function of Discourse-Topic Structure

This extension introduces **discourse-topic structure** (Jubran, 2015 — *tópico discursivo*) as an
independent annotation layer over the Roda Viva benchmark, allowing the calibration audit to ask not
only *whether* the model ignores epistemic cues, but **where** in the topical organization of the talk
its miscalibration concentrates.

### Why this layer

The original pilot risked **circularity**: the human certainty score penalizes hesitation markers, and
the finding was that the model ignores hesitation markers. Discourse-topic structure is derived from an
independent criterion (topical *centração* + *organicidade*) and breaks that circularity, providing a
theory-grounded unit of analysis larger than the turn.

### What was added

- **Annotated benchmark** (`data/stil_benchmark_topico_discursivo.csv`): 344 turns across 3 interviews,
  segmented into **81 Discourse-Topic Frames (Quadros Tópicos)**, with per-turn position
  (opening / development / resumption / digression / framing / closing), topical-boundary and
  post-break flags, a `ZONA` field, and hesitation-marker counts (density per 100 words).
- **Annotation script** (`src/anotar_topico.py`): reproduces the topic columns + hesitation density.
- **Calibration-by-topic script** (`src/ece_por_zona.py`): ECE by topical zone, by hesitation tercile,
  and global / post-break.
- **Audit pipeline** (`notebooks/02_audit_llama_topico.py`): Llama-3.1-8B-Instruct confidence via a
  **normalized Yes/No** probability (a single-token probability collapses to ~0 and is unusable).

### Key findings (Llama-3.1-8B-Instruct, 291 content turns)

- The model's confidence is **decoupled** from human certainty (r = 0.10) and sits flat around 40–50.
- Miscalibration is **structured by topic**: largest at topic openings (gap −25.0 vs −9.7 internally),
  in low-hesitation/assertive topics (−26.0), and for the most categorical speaker (−24.5); near zero
  for the most tentative speaker (−1.1).
- Reframing: the risk is not blind **overconfidence** but **structured insensitivity** — the model fails
  to recognize certainty where it is strongest.

### Reproduce

```bash
# 1. (Colab, GPU) run the audit -> final_audit_results.csv
python notebooks/02_audit_llama_topico.py

# 2. calibration broken down by discourse-topic structure
python src/ece_por_zona.py --anotado results/final_audit_results.csv
```

### References

Jubran (2015), *Tópico discursivo*, in *Gramática do português culto falado no Brasil*, v. II;
Marcuschi (2006), *A hesitação*, idem v. I; Guo et al. (2017), *On calibration of modern neural networks*, ICML;
Hyland (2005), *Metadiscourse*.
