# Uncertainty Signature Audit

**Speech Disfluencies and LLM Calibration: Auditing Probabilistic 
Overconfidence through Pragmatic Markers in Spontaneous Speech**

This repository contains the experimental framework developed to 
investigate how the suppression of pragmatic markers (hesitations, 
fillers, and hedges) affects the probabilistic calibration of Large 
Language Models (LLMs).

## Research Overview

This project introduces the concept of **pragmatic blindness** — 
the systematic failure of LLMs to process oral planning markers as 
epistemic uncertainty signals — and provides empirical evidence 
through a calibration audit on the Roda Viva Corpus (Brazilian 
Portuguese).

A paper based on this work has been submitted to the **2nd Joint 
Workshop on Discourse and Dialogue (CODI-CRAC 2026)**, co-located 
with ACL 2026 (under review).

## Key Findings

- **Dataset:** 344 contrastive turns from the Roda Viva corpus 
  (3 interviews: Heloísa Starling, Marco Aurélio Mello, Galvão Bueno)
- **Model:** Meta-Llama-3.1-8B-Instruct (4-bit quantization)
- **ECE:** Layer A (Faithful) = 41.95 | Layer B (Sanitized) = 41.14
- **OE:** Layer A = 4.29 | Layer B = 3.31
- **Wilcoxon test:** W = 10988.50, p = 0.0023
- **Spearman correlation:** ρ = −0.49 (faithful), ρ = −0.43 (sanitized)

## Epistemic Commitment Annotation Matrix

| Category | Marker | Example | Points | Theoretical Basis |
|---|---|---|---|---|
| Epistemic Hedges | Lexical hedges | maybe, I think, seems | −15 | Epistemic retreat (Hyland, 2005) |
| Reformulations | False starts | has to end... rewrite | −10 | Syntactic abandonment (Marcuschi, 2003) |
| Filled Pauses | Hesitation vocalizations | uh..., um... | −5 | Macrostructural planning marker |
| Lengthenings | Vowel prolongation | veryyy, forrrr | −5 | Lexical selection in progress |
| Repetitions | Term repetition | that that, but but | −5 | Rhythmic hesitation |

## Repository Structure
```
/data
    pilot_benchmark.csv     # 344-turn contrastive dataset
/notebooks
    02_llm_probabilistic_audit.ipynb  # Logit extraction and ECE/OE pipeline
/results
    final_audit_results.csv  # AI confidence scores per turn
    calibration_plot.png     # Reliability Diagram
/src
    audit_pipeline.py        # Main pipeline script
/docs
    annotation_guide.md      # Epistemic commitment annotation protocol
```

## Methodology

The experiment contrasts two transcription conditions:

- **Layer A (Faithful):** Jeffersonian conventions preserved — 
  micropauses `(.)`, lengthenings `(::)`, truncations, filled pauses
- **Layer B (Sanitized):** disfluency markers suppressed, 
  approximating written standard norms

Model confidence was extracted via Softmax on the YES token logit 
for the binary prompt: *"Based strictly on the text, does the 
speaker express absolute conviction about the information? 
Answer only YES or NO."*

## Tech Stack

- Python (PyTorch, Hugging Face Transformers)
- Meta-Llama-3.1-8B-Instruct (4-bit quantization)
- SciPy (Wilcoxon, Spearman)
- Matplotlib, Seaborn

## Citation

If you use this work, please cite:
```
Santos, V. V. (2026). Speech Disfluencies and LLM Calibration: 
A Pilot Study on the Effects of Textual Sanitization in 
Brazilian Portuguese. Submitted to CODI-CRAC 2026 (ACL Workshop).
```

## References

- Marcuschi, L. A. (2003). *Análise da Conversação*. Ática.
- Hyland, K. (2005). *Metadiscourse*. Continuum.
- Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.
- Vale et al. (2024). Roda Viva Corpus. PROPOR 2024.

## License

MIT License — developed as part of PhD research at the Federal 
University of São Carlos (UFSCar), Brazil.
```

scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
numpy>=1.24.0
