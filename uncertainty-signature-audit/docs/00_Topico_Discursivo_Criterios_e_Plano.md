# Tópico discursivo × auditoria de calibração de LLM — critérios de anotação e plano do ensaio

Disciplina: *Fenômenos de construção textual-interativa* — ensaio final
Articulação: organização tópica (Jubran, 2015) aplicada ao pilot **uncertainty-signature-audit** (Corpus Roda Viva)

> **Status do CSV:** primeira passada (rascunho) gerada automaticamente a partir do conteúdo dos turnos. **A segmentação precisa da sua revisão** — o julgamento interpretativo da fronteira tópica é a sua contribuição autoral e não pode ser terceirizado. Use este documento para validar ou corrigir.

---

## 1. Por que tópico discursivo (e não "trocar o proxy")

O tópico discursivo **não mede certeza epistêmica**; descreve a *organização* do texto falado por duas propriedades (Jubran, 2015):

- **Centração** — concernência, relevância e pontualização: o conjunto de enunciados gravita em torno de um foco referencial localizável.
- **Organicidade** — relações de interdependência nos planos hierárquico (vertical) e linear (horizontal).

Por isso ele entra como **camada moderadora independente**, não como substituto da `NOTA_HUMANA`. Isso resolve a crítica de **circularidade** do pilot atual (a nota penaliza marcas de hesitação e depois "descobre" que o modelo as ignora): a segmentação tópica é derivada de critério próprio, alheio à contagem de hesitação.

A pergunta de pesquisa do ensaio passa a ser:

> A hesitação humana e a superconfiança do LLM se distribuem de modo homogêneo na estrutura tópica, ou se concentram em zonas específicas (fronteiras, retomadas pós-intervalo, ou determinados Quadros Tópicos)?

---

## 2. Unidades e critério operacional

- **Quadro Tópico (QT):** bloco de turnos com a mesma centração (mesmo foco temático).
- **Posição do turno no QT:**
  - `abertura` — turno que introduz o QT (em entrevista, quase sempre o entrevistador).
  - `desenvolvimento` — turnos que mantêm/elaboram o QT (em entrevista, o núcleo do entrevistado).
  - `retomada` — reabertura de tópico após corte/digressão (tipicamente após intervalo).
  - `digressao` — inserção fora da centração corrente (banter, comentário lateral).
  - `moldura` — gestão metadiscursiva do programa (passar a palavra, anunciar intervalo).
  - `backchannel` — turnos mínimos de monitoramento ("mm hum", "uhum").
  - `fechamento` — encerramento.
- **`FRONTEIRA_TOPICA` = 1** para abertura/retomada/fechamento (junturas).
- **`POS_INTERVALO` = 1** no 1º turno de conteúdo após corte comercial (descontinuidade forte).
- **`ZONA`** (variável de cruzamento com ECE): `fronteira` / `interior` / `moldura` / `backchannel` / `digressao`.

---

## 3. Colunas adicionadas ao CSV

| Coluna | Conteúdo |
|---|---|
| `QT_ID` | identificador do QT (ex.: `MA-07`, `HS-05`) |
| `QUADRO_TOPICO` | rótulo temático (centração) |
| `POSICAO_TOPICA` | abertura / desenvolvimento / retomada / digressao / moldura / backchannel / fechamento |
| `FRONTEIRA_TOPICA` | 1 = juntura tópica |
| `POS_INTERVALO` | 1 = 1º turno de conteúdo pós-intervalo |
| `ZONA` | fronteira / interior / moldura / backchannel / digressao |
| `N_MICROPAUSA` | nº de `(.)` |
| `N_PAUSA_LONGA` | nº de `(..)` |
| `N_TRUNCAMENTO` | nº de truncamentos/reformulações (`palavra-`) |
| `N_ALONGAMENTO` | nº de `::` |
| `N_PREENCHIDA` | nº de pausas preenchidas (ééé, ahn, hum, eh, uhum...) |
| `N_PALAVRAS` | palavras na `FALA_FIEL` |
| `DENS_HESIT_100` | marcas de hesitação por 100 palavras |

Segmentação: **29 QTs** (Marco Aurélio), **29 QTs** (Galvão), **23 QTs** (Heloisa).

---

## 4. Achados preliminares (já rodados — servem de espinha dorsal empírica)

**(a) Validação cruzada da anotação.** Correlação `DENS_HESIT_100 × NOTA_HUMANA = −0,384` (n=291): mais hesitação → menor certeza atribuída pelo humano. A direção é a esperada e a magnitude é *moderada* — ou seja, a nota não é um mero recodificador da contagem de marcas (não é circular), o que **fortalece** o uso da nota como referência.

**(b) A hesitação não se concentra na fronteira — concentra-se no desenvolvimento.** Nas falas do entrevistado (≥8 palavras):

- `interior` (desenvolvimento): média **6,84** marcas/100 palavras (n=132)
- `fronteira`: média **5,03** (n=8)

Isto **nuança** a hipótese inicial de "superconfiança na fronteira". No gênero entrevista, quem abre o tópico é o entrevistador; o entrevistado hesita ao **desenvolver** o conteúdo (esforço de planejamento e monitoramento, Marcuschi). A unidade onde mora a incerteza epistêmica é o *interior* do tópico. **Este é um bom achado para o ensaio**, não um problema: mostra que a posição na estrutura tópica condiciona onde a hesitação emerge.

**(c) A hesitação varia por Quadro Tópico.** Os QTs mais densos são os de Heloisa Starling (memória da ditadura, corrupção, democracia — temas epistêmica e afetivamente custosos); os menos densos são os de Galvão (anedóticos, asseverativos). Isso sugere um eixo de análise: **a calibração do LLM acompanha a "dificuldade tópica"?** Hipótese: ECE maior nos QTs de alta densidade de hesitação.

---

## 5. O que rodar agora (seus scripts)

Sobre `stil_benchmark_topico_discursivo.csv`, reaproveitando o pipeline de logits/ECE do repositório:

1. **ECE por `ZONA`** — comparar erro de calibração em `fronteira` vs `interior`. Espera-se, à luz de (b), que o desalinhamento seja maior no `interior` (onde o humano hesita e o modelo, lendo a versão higienizada, asservera).
2. **ECE por densidade de hesitação do QT** — agrupar QTs em tercis de `DENS_HESIT_100` e medir ECE por tercil (testa a hipótese (c)).
3. **Retomadas pós-intervalo** (`POS_INTERVALO=1`) — inspeção qualitativa: o modelo "reinicia" o estado epistêmico a cada bloco?
4. Excluir `moldura` e `backchannel` das análises de calibração (não carregam asserção de conteúdo).

---

## 6. Esqueleto do ensaio

1. **Introdução** — o enunciado da disciplina e a tese: a organização tópica é um fenômeno de construção textual-interativa que pode servir de *unidade de análise* para auditar a calibração de LLMs.
2. **Fundamentação** — tópico discursivo (Jubran, 2015: centração + organicidade); hesitação como sintaxe de planejamento/monitoramento (Marcuschi); calibração e ECE (Guo et al., 2017).
3. **Articulação com a pesquisa** — do proxy de certeza à camada tópica; como isso quebra a circularidade.
4. **Ilustração empírica** — as 3 entrevistas do Roda Viva, segmentação tópica, achados (a)–(c).
5. **Discussão** — implicações para *AI safety*: o modelo erra de forma estruturada em relação à organização do texto falado.
6. **Limitações** — anotador único, sem cálculo de concordância (kappa); piloto demonstrativo; segmentação revisável.
7. **Conclusão** — agenda: anotação validada por 2º anotador, expansão do corpus, e medida formal por QT.

---

## 7. Limitações a declarar (honestidade metodológica)

- Anotação de **um único anotador**; sem kappa de concordância — adequado a um ensaio/piloto, não a um paper.
- Segmentação tópica em entrevista jornalística: as aberturas são quase sempre do entrevistador, o que reduz o n de fronteiras "do entrevistado" — discutir como limite de gênero.
- Contagem de marcas de hesitação é heurística (regex); a `NOTA_HUMANA` permanece como referência qualitativa.
