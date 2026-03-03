# uncertainty-signature-audit
# The Uncertainty Signature: Auditing LLM Calibration through Pragmatic Markers

This repository contains the auditing framework developed to analyze how the suppression of pragmatic markers (hesitations, fillers, and hedges) affects the reliability and calibration of Large Language Models (LLMs).

## Research Objective
The core of this project is to investigate the correlation between the "cleaning" or "sanitization" of spoken language transcripts—specifically from the **Roda Viva Corpus** (Brazilian Portuguese)—and the emergence of **Overconfidence Hallucinations** in models like Llama-3 and GPT-4.

By removing human hesitation, we hypothesize that LLMs are forced into a state of categorical certainty that does not reflect the speaker's original epistemic stance.

## Theoretical Framework
This interdisciplinary framework bridges Linguistics and AI Safety:
* **Text Linguistics:** Syntax of planning and monitoring (Marcuschi, 2001).
* **Pragmatics:** Interpersonal metadiscourse and Hedges (Hyland, 2005).
* **AI Safety/Machine Learning:** Calibration of neural networks and Expected Calibration Error (Guo et al., 2017).

## Transcription Protocol (Adapted Jeffersonian Notation)
For uncertainty analysis, we preserve the following markers to capture the speaker's "uncertainty signature":
* `::` Vowel elongation (indicates lexical search).
* `(.)` Micropause (indicates syntactic planning).
* `uh / um / eh` Filled pauses (monitoring markers).
* `-` Truncation or sudden reformulation.

## Methodology & Tech Stack
1.  **Data:** Selected excerpts from the Roda Viva interviews (focusing on complex socio-political discourse).
2.  **Audit:** Comparing Logit/Softmax distributions between "Sanitized" vs. "Pragmatic-Rich" inputs.
3.  **Metrics:** Calculation of ECE (Expected Calibration Error).
4.  **Tools:** Python, PyTorch, Hugging Face Transformers.

*Developed as part of a PhD research project at the Federal University of São Carlos (UFSCar), Brazil.*
