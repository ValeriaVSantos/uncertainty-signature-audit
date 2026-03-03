#The Uncertainty Signature: Auditing LLM Calibration through Pragmatic Markers

This repository contains the auditing framework developed to analyze how the suppression of pragmatic markers (hesitations, fillers, and hedges) affects the reliability and calibration of Large Language Models (LLMs).

#Research Objective

The core of this project is to investigate the correlation between the "cleaning" or "sanitization" of spoken language transcripts—specifically from the Roda Viva Corpus (Brazilian Portuguese)—and the emergence of Overconfidence Hallucinations in models like Llama-3 and GPT-4.

By removing human hesitation, we hypothesize that LLMs are forced into a state of categorical certainty that does not reflect the speaker's original epistemic stance.

Theoretical Framework

This interdisciplinary framework bridges Linguistics and AI Safety:

Text Linguistics: Syntax of planning and monitoring (Marcuschi, 2001).

Pragmatics: Interpersonal metadiscourse and Hedges (Hyland, 2005).

AI Safety/Machine Learning: Calibration of neural networks and Expected Calibration Error (Guo et al., 2017).

Transcription Protocol (Adapted Jeffersonian Notation)

For uncertainty analysis, we preserve the following markers to capture the speaker's "uncertainty signature":

:: Vowel elongation (indicates lexical search).

(.) Micropause (indicates syntactic planning).

uh / um / eh Filled pauses (monitoring markers).

- Truncation or sudden reformulation.

📊 Empirical Audit: Llama 3.1 Results

The first technical audit was performed using the Meta-Llama-3.1-8B-Instruct model to measure its sensitivity to the Jeffersonian markers listed above.

Key Findings

Model: Llama-3.1-8B-Instruct (4-bit quantization).

Dataset: 95 sanitized vs. pragmatic-rich excerpts from the Heloisa Starling interview.

Expected Calibration Error (ECE): 52.88%.

Analysis: The high ECE score indicates significant miscalibration. The model consistently assigns high confidence (overconfidence) to statements that humans, guided by pragmatic markers, identify as hesitant or uncertain. This suggests that LLMs "blindly" process sanitized text, ignoring the epistemic nuances preserved by the Jeffersonian notation.

Repository Structure

/data:

pilot_benchmark.csv: The primary dataset (95 rows) with human ground-truth scores.

/notebooks:

02_llm_probabilistic_audit.ipynb: Python pipeline for logit extraction and ECE calculation.

/results:

final_audit_results.csv: Audit output including AI confidence scores.

calibration_plot.png: Reliability Diagram showing the gap between human perception and AI confidence.

Methodology & Tech Stack

Data: Selected excerpts from the Roda Viva interviews (focusing on complex socio-political discourse).

Audit: Comparing Logit/Softmax distributions for "Absolute Certainty" queries.

Metrics: ECE (Expected Calibration Error).

Tools: Python, PyTorch, Hugging Face Transformers.

Developed as part of a PhD research project at the Federal University of São Carlos (UFSCar), Brazil.
