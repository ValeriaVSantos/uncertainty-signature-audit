#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECE por ZONA TOPICA e por TERCIL DE HESITACAO
=============================================
Cruza a saida da auditoria de logits do repositorio uncertainty-signature-audit
com a anotacao de topico discursivo (stil_benchmark_topico_discursivo.csv).

USO
---
1) Rode seu pipeline (02_llm_probabilistic_audit.ipynb) sobre o CSV anotado e
   salve um arquivo com, no minimo, as colunas:
        ID            -> mesmo ID do benchmark
        AI_CONF       -> confianca do modelo em [0,1] (prob. de "certeza absoluta")
   (Se sua coluna tiver outro nome, ajuste COL_CONF abaixo ou passe --conf NOME.)

2) Rode:
        python ece_por_zona.py \
            --anotado stil_benchmark_topico_discursivo.csv \
            --audit   final_audit_results.csv \
            --conf    AI_CONF \
            --out     ece_por_zona_resultados.csv

Sem --audit, o script roda em modo DESCRITIVO (so densidade de hesitacao por zona),
util para conferir a estrutura antes de ter os logits.

DEFINICOES
----------
- Referencia humana de certeza: NOTA_HUMANA / 100  (em [0,1]).
- Gap de calibracao por turno: |AI_CONF - NOTA_HUMANA/100|.
- ECE (binned): erro de calibracao esperado com M bins de confianca iguais
  (Guo et al., 2017), usando a referencia humana como "accuracy alvo".
"""
import argparse, sys
import numpy as np
import pandas as pd

COL_ID   = "ID"
COL_NOTA = "NOTA_HUMANA"
COL_CONF = "CONFIANCA_IA"   # nome gerado pelo notebook 02_llm_probabilistic_audit
MIN_PALAVRAS = 8            # exclui turnos muito curtos (backchannels)
ZONAS_ANALISE = ["fronteira", "interior"]   # foco; moldura/backchannel/digressao ficam de fora

def ece_binned(conf, target, n_bins=10):
    """ECE com bins de confianca de largura igual. target em [0,1]."""
    conf = np.asarray(conf, float); target = np.asarray(target, float)
    if len(conf) == 0: return np.nan
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0: continue
        ece += (m.sum()/n) * abs(conf[m].mean() - target[m].mean())
    return ece

def gap_medio(conf, target):
    conf = np.asarray(conf, float); target = np.asarray(target, float)
    return np.nan if len(conf) == 0 else float(np.mean(np.abs(conf - target)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anotado", default="stil_benchmark_topico_discursivo.csv")
    ap.add_argument("--audit", default=None, help="CSV da auditoria com ID + confianca do modelo")
    ap.add_argument("--conf", default=COL_CONF, help="nome da coluna de confianca no --audit")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--out", default="ece_por_zona_resultados.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.anotado)
    df = df[df["N_PALAVRAS"].astype(int) >= MIN_PALAVRAS].copy()

    # ---------- modo descritivo (sem logits) ----------
    print("="*64)
    print("DENSIDADE DE HESITACAO POR ZONA (todas as falas, >= %d palavras)" % MIN_PALAVRAS)
    print("="*64)
    g = df[df["ZONA"].isin(ZONAS_ANALISE)].groupby("ZONA")["DENS_HESIT_100"]
    print(g.agg(["count", "mean", "median"]).round(2).to_string())

    # ---------- obter a confianca do modelo ----------
    # Caso 1: o proprio CSV anotado ja tem a coluna (voce auditou o CSV anotado).
    # Caso 2: a confianca esta num arquivo separado -> --audit, merge por ID.
    if args.conf in df.columns:
        print(f"\n[ok] coluna '{args.conf}' encontrada no proprio CSV anotado.")
    elif args.audit:
        aud = pd.read_csv(args.audit)
        if args.conf not in aud.columns:
            sys.exit(f"ERRO: coluna '{args.conf}' nao existe em {args.audit}. "
                     f"Colunas disponiveis: {list(aud.columns)}")
        df = df.merge(aud[[COL_ID, args.conf]], on=COL_ID, how="inner")
        print(f"\nMerge: {len(df)} turnos com confianca do modelo.")
    else:
        print("\n[modo descritivo] Sem coluna de confianca e sem --audit: pulei o ECE.")
        print(f"Rode a auditoria (gera '{args.conf}') e rode de novo.")
        return

    df["AI_CONF"] = df[args.conf].astype(float)
    # auto-escala: o notebook salva CONFIANCA_IA em 0-100; aqui usamos 0-1.
    if df["AI_CONF"].max() > 1.5:
        df["AI_CONF"] = df["AI_CONF"] / 100.0
        print("  (confianca normalizada de 0-100 para 0-1)")
    df["TARGET"] = df[COL_NOTA].astype(float) / 100.0
    df["GAP"] = (df["AI_CONF"] - df["TARGET"]).abs()

    rows = []

    # H1: ECE por ZONA (fronteira vs interior)
    print("\n" + "="*64)
    print("H1 — ECE POR ZONA TOPICA")
    print("="*64)
    for z in ZONAS_ANALISE:
        sub = df[df["ZONA"] == z]
        e = ece_binned(sub["AI_CONF"], sub["TARGET"], args.bins)
        gm = gap_medio(sub["AI_CONF"], sub["TARGET"])
        print(f"  {z:10s}: n={len(sub):3d}  ECE={e:.4f}  gap_medio={gm:.4f}")
        rows.append({"analise": "zona", "grupo": z, "n": len(sub),
                     "ECE": round(e,4), "gap_medio": round(gm,4)})

    # H2: ECE por TERCIL de densidade de hesitacao (no nivel do QT)
    print("\n" + "="*64)
    print("H2 — ECE POR TERCIL DE DENSIDADE DE HESITACAO (por Quadro Topico)")
    print("="*64)
    qt = df.groupby("QT_ID")["DENS_HESIT_100"].mean()
    try:
        terc = pd.qcut(qt, 3, labels=["baixa", "media", "alta"])
    except ValueError:
        terc = pd.cut(qt, 3, labels=["baixa", "media", "alta"])
    df["TERCIL_QT"] = df["QT_ID"].map(terc)
    for tlab in ["baixa", "media", "alta"]:
        sub = df[df["TERCIL_QT"] == tlab]
        if len(sub) == 0: continue
        e = ece_binned(sub["AI_CONF"], sub["TARGET"], args.bins)
        gm = gap_medio(sub["AI_CONF"], sub["TARGET"])
        print(f"  {tlab:6s}: n={len(sub):3d}  ECE={e:.4f}  gap_medio={gm:.4f}")
        rows.append({"analise": "tercil_hesit", "grupo": tlab, "n": len(sub),
                     "ECE": round(e,4), "gap_medio": round(gm,4)})

    # H3: ECE global e em retomadas pos-intervalo
    print("\n" + "="*64)
    print("H3 — GLOBAL e RETOMADAS POS-INTERVALO")
    print("="*64)
    for nome, sub in [("global", df), ("pos_intervalo", df[df["POS_INTERVALO"]==1])]:
        e = ece_binned(sub["AI_CONF"], sub["TARGET"], args.bins)
        gm = gap_medio(sub["AI_CONF"], sub["TARGET"])
        print(f"  {nome:14s}: n={len(sub):3d}  ECE={e:.4f}  gap_medio={gm:.4f}")
        rows.append({"analise": "geral", "grupo": nome, "n": len(sub),
                     "ECE": round(e,4), "gap_medio": round(gm,4)})

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nResultados salvos em: {args.out}")

if __name__ == "__main__":
    main()
