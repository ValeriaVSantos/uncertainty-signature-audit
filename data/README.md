# Data documentation

## Main benchmark

`stil_pilot_benchmark_consolidado.csv` contains the 344 contrastive turns used in the published study.

| Column | Description |
|---|---|
| `ID` | Turn identifier |
| `ENTREVISTA` | Interview identifier |
| `LOCUTOR` | Speaker |
| `FALA_FIEL` | Faithful transcript preserving disfluency markers |
| `FALA_HIGIENIZADA` | Sanitized version of the same turn |
| `NOTA_HUMANA` | Deductive epistemic-commitment proxy on a 0-100 scale |

`stil_benchmark_topico_discursivo.csv` extends the table with discourse-topic and marker-count fields used in subsequent exploratory analysis.

The three `pilot_benchmark*.csv` files preserve the earlier per-interview workflow and are not the recommended entry point for reproducing the published 344-turn analysis.

## Source and reuse

The turns derive from the [LeGOS-UFSCar Roda Viva corpus](https://github.com/LeGOS-UFSCar/Roda-Viva). Consult the upstream corpus terms before redistributing transcript content. The MIT License at the repository root applies to code, not automatically to third-party source text.
