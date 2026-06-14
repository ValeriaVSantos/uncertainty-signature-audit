# -*- coding: utf-8 -*-
"""
Anotacao de TOPICO DISCURSIVO (Jubran, 2015) sobre o benchmark do pilot
uncertainty-signature-audit. Primeira passada (rascunho) para revisao humana.

Adiciona ao CSV original as colunas:
  QT_ID            identificador do Quadro Topico (ex.: MA-07)
  QUADRO_TOPICO    rotulo tematico do QT (centracao)
  POSICAO_TOPICA   abertura / desenvolvimento / retomada / digressao / moldura / backchannel / fechamento
  FRONTEIRA_TOPICA 1 se o turno esta em juntura topica (abertura/retomada/fechamento)
  POS_INTERVALO    1 se e o 1o turno de conteudo apos corte comercial (descontinuidade forte)
  ZONA             fronteira / interior / moldura / backchannel / digressao  (variavel para cruzar com ECE)
  N_MICROPAUSA     # de (.)
  N_PAUSA_LONGA    # de (..)
  N_TRUNCAMENTO    # de truncamentos/reformulacoes (palavra-)
  N_ALONGAMENTO    # de :: (alongamento vocalico)
  N_PREENCHIDA     # de pausas preenchidas (ééé, ahn, hum, eh, uhum...)
  N_PALAVRAS       # de palavras na FALA_FIEL
  DENS_HESIT_100   marcas de hesitacao por 100 palavras (micropausa+longa+trunc+along+preench)
"""
import csv, re

SRC = "/sessions/serene-relaxed-goodall/mnt/uploads/stil_pilot_benchmark_consolidado (1).csv"
OUT = "/sessions/serene-relaxed-goodall/mnt/outputs/stil_benchmark_topico_discursivo.csv"

# ---------- 1. faixas de Quadro Topico (start_id, end_id, QT_ID, rotulo) ----------
QT_RANGES = [
    # ===== MARCO AURELIO (1-126) =====
    (1, 4,   "MA-01", "Abertura do programa e divulgacao das gravacoes (Lava Jato)"),
    (5, 6,   "MA-02", "Eleicoes gerais e renuncia coletiva"),
    (7, 10,  "MA-03", "Foro privilegiado / prerrogativa de foro"),
    (11, 12, "MA-04", "Impeachment: golpe ou legalidade constitucional"),
    (13, 16, "MA-05", "Supremo 'acovardado' e independencia do juiz"),
    (17, 20, "MA-06", "Lula, foro e persecucao criminal"),
    (21, 27, "MA-07", "Leniencia do Supremo / caso Eduardo Cunha"),
    (28, 29, "MA-08", "Imparcialidade do ministro / parte interessada"),
    (30, 31, "MA-09", "Rito do impeachment na Camara"),
    (32, 39, "MA-10", "Obstrucao de justica / arquivamento por Cunha"),
    (40, 41, "MA-11", "Publicidade x sigilo / TV Justica"),
    (42, 48, "MA-12", "Articulacoes politicas e impeachment de Temer"),
    (49, 59, "MA-13", "Evolucao institucional e otimismo (debate com Neumanne)"),
    (60, 66, "MA-14", "Policia Federal / autonomia / Anjo Aragao"),
    (67, 71, "MA-15", "Morosidade / habeas corpus (Celso Daniel)"),
    (72, 73, "MA-16", "Atualidade das ideias de 2006"),
    (74, 75, "MA-17", "Epidemias e politicas publicas"),
    (76, 80, "MA-18", "Processo eleitoral da chapa Dilma-Temer (TSE)"),
    (81, 86, "MA-19", "Caso Barroso / critica ao PMDB"),
    (87, 88, "MA-20", "Poder moderador e contaminacao dos poderes"),
    (89, 97, "MA-21", "Recurso ao Supremo no impeachment"),
    (98, 103,"MA-22", "Lula na Casa Civil / desvio de finalidade"),
    (104,105,"MA-23", "Manobra de foro / renuncia como fraude"),
    (106,109,"MA-24", "Ataques pessoais e familia"),
    (110,110,"MA-25", "Fenomeno Sergio Moro"),
    (111,116,"MA-26", "Capa do NYT / delacao premiada"),
    (117,121,"MA-27", "Balanco: instituicoes funcionando / Lava Jato"),
    (122,123,"MA-28", "Intolerancia e polarizacao nas redes"),
    (124,126,"MA-29", "Encerramento do programa"),
    # ===== GALVAO BUENO (127-249) =====
    (127,128,"GB-00", "Abertura do programa e saudacoes"),
    (129,130,"GB-01", "Aposentadoria e atividades atuais"),
    (131,133,"GB-02", "Imagem publica / influenciador / memes"),
    (134,138,"GB-03", "Olimpiadas: narrar a distancia (Toquio)"),
    (139,141,"GB-04", "Relacao com idolos / Neymar"),
    (142,148,"GB-05", "Narracao dos titulos: tetra e penta"),
    (149,159,"GB-06", "Saida da Globo e fim do narrador (2022)"),
    (160,164,"GB-07", "Mensagem de Reginaldo Leme / amizade na F1"),
    (165,169,"GB-08", "Formula 1: inicio em 1980 (Bandeirantes)"),
    (170,172,"GB-09", "Ayrton Senna / luto"),
    (173,175,"GB-10", "Tetra de 1994 / bastidores"),
    (176,178,"GB-11", "Primeira maratona feminina olimpica"),
    (179,182,"GB-12", "Saida da Globo nos anos 90 / TVA"),
    (183,187,"GB-13", "Posicionamento social / racismo / voz publica"),
    (188,190,"GB-14", "Trajetoria na Globo anos 80 / Luciano do Valle"),
    (191,195,"GB-15", "Imparcialidade / 'qual o time do Galvao'"),
    (196,201,"GB-16", "Saude / infarto na final da Libertadores 2019"),
    (202,209,"GB-17", "Mercado de transmissao / streaming"),
    (210,213,"GB-18", "Abuso e responsabilidade no futebol"),
    (214,216,"GB-19", "7x1 e Copa 2014 / Felipao"),
    (217,221,"GB-20", "Canal proprio e planos"),
    (222,225,"GB-21", "Emocao das narracoes: 94 e 2002 (revisita)"),
    (226,229,"GB-22", "Tragedia no RS / enchentes / posicionamento"),
    (230,236,"GB-23", "Mulheres na locucao esportiva / machismo"),
    (237,238,"GB-24", "Livro e definicao de Senna (revisita)"),
    (239,240,"GB-25", "Juventude / ditadura / democracia"),
    (241,243,"GB-26", "Inicio de carreira / Darcy Reis"),
    (244,247,"GB-27", "Bordoes e espontaneidade"),
    (248,249,"GB-28", "Encerramento do programa"),
    # ===== HELOISA STARLING (250-344) =====
    (250,251,"HS-01", "Abertura / decisao de Lula e memoria do golpe"),
    (252,255,"HS-02", "Minas Gerais e o golpe de 64"),
    (256,257,"HS-03", "Papel dos governadores no golpe"),
    (258,264,"HS-04", "Mourao (1937/1964) e continuidades"),
    (265,272,"HS-05", "Tutela militar / artigo 142"),
    (273,275,"HS-06", "Sociedade civil e memoria (retomada)"),
    (276,280,"HS-07", "Anatomia do golpe / etapas"),
    (281,285,"HS-08", "Relevancia hoje / o garoto da padaria"),
    (286,288,"HS-09", "Colaboracionismo civil / empresarios"),
    (289,291,"HS-10", "Estrutura social / latifundio / ligas camponesas"),
    (292,295,"HS-11", "Ditadura e populacao negra / memoria"),
    (296,298,"HS-12", "Bolsonaro e autoritarismo contemporaneo"),
    (299,304,"HS-13", "Avalistas do golpe / comunicacao e apoio popular"),
    (305,309,"HS-14", "Cultura democratica e defesa da democracia"),
    (310,312,"HS-15", "Militarizacao da policia / herancas"),
    (313,315,"HS-16", "Direito a memoria / resistencia cultural"),
    (316,318,"HS-17", "Mitos da ditadura / corrupcao"),
    (319,323,"HS-18", "Mulheres na conspiracao / marcha da familia"),
    (324,328,"HS-19", "CLT e direitos trabalhistas hoje"),
    (329,330,"HS-20", "Estado da democracia brasileira hoje"),
    (331,334,"HS-21", "Historia publica / papel do historiador"),
    (335,338,"HS-22", "Avaliacao do governo / PT"),
    (339,344,"HS-23", "Forcas democraticas / encerramento"),
]

# ---------- 2. posicoes que nao sao 'desenvolvimento' (default) ----------
ABERTURA = {1,5,7,11,13,17,21,30,32,40,43,49,60,67,74,76,81,87,89,104,106,110,111,117,122,
            129,132,135,140,143,152,162,166,170,174,177,179,189,192,196,202,215,218,222,231,237,239,242,245,
            250,253,256,259,265,277,282,287,290,292,299,306,311,314,316,320,325,329,331,335,340}
RETOMADA = {28,72,98, 161,184,211,227, 274,297}
MOLDURA  = {39,42,66,71,73,80,97,99,126,
            127,131,134,139,142,149,150,160,163,165,173,176,188,191,210,214,217,226,230,241,244,
            252,273,276,281,286,289,296,305,310,313,319,324,339,344}
DIGRESSAO= {36, 128,151,247, 258}
FECHAMENTO={124,125, 248, 343}
BACKCHANNEL={147,154,158,186,194,200,206,208,212,224,
             260,263,266,271,283,294,301,308,322,327,333,342}
POS_INTERVALO={28,72,98, 161,184,211,227, 274,297,314,325}

def posicao(i):
    if i in FECHAMENTO: return "fechamento"
    if i in ABERTURA:   return "abertura"
    if i in RETOMADA:   return "retomada"
    if i in MOLDURA:    return "moldura"
    if i in DIGRESSAO:  return "digressao"
    if i in BACKCHANNEL:return "backchannel"
    return "desenvolvimento"

def zona(pos, posint):
    if pos in ("abertura","retomada","fechamento") or posint:
        return "fronteira"
    if pos == "moldura":    return "moldura"
    if pos == "backchannel":return "backchannel"
    if pos == "digressao":  return "digressao"
    return "interior"

def qt_of(i):
    for a,b,qid,lab in QT_RANGES:
        if a <= i <= b: return qid, lab
    return "NA","NA"

# ---------- 3. contagem de marcas de hesitacao (sobre FALA_FIEL) ----------
RE_PREENCH = re.compile(r'\b(?:é{2,}|ah+n|ah+m|hum+|eh+|uh+|um|aham|ahã|ehn)\b', re.IGNORECASE)
RE_TRUNC   = re.compile(r'[A-Za-zÀ-ÿ]{1,}-(?=\s|$|[\.,\?\)])')

def conta(txt):
    micro = txt.count("(.)")
    longa = txt.count("(..)")
    along = txt.count("::")
    trunc = len(RE_TRUNC.findall(txt))
    preen = len(RE_PREENCH.findall(txt))
    palavras = len(re.findall(r'\w+', txt))
    total = micro + longa + along + trunc + preen
    dens = round(100.0*total/palavras, 2) if palavras else 0.0
    return micro, longa, trunc, along, preen, palavras, dens

# ---------- 4. gerar ----------
rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
fields = list(rows[0].keys()) + ["QT_ID","QUADRO_TOPICO","POSICAO_TOPICA","FRONTEIRA_TOPICA",
    "POS_INTERVALO","ZONA","N_MICROPAUSA","N_PAUSA_LONGA","N_TRUNCAMENTO","N_ALONGAMENTO",
    "N_PREENCHIDA","N_PALAVRAS","DENS_HESIT_100"]

with open(OUT,"w",encoding="utf-8",newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for r in rows:
        i = int(r["ID"])
        pos = posicao(i)
        posint = 1 if i in POS_INTERVALO else 0
        qid, lab = qt_of(i)
        micro,longa,trunc,along,preen,pal,dens = conta(r["FALA_FIEL"])
        r.update({
            "QT_ID":qid,"QUADRO_TOPICO":lab,"POSICAO_TOPICA":pos,
            "FRONTEIRA_TOPICA":1 if pos in ("abertura","retomada","fechamento") else 0,
            "POS_INTERVALO":posint,"ZONA":zona(pos,posint),
            "N_MICROPAUSA":micro,"N_PAUSA_LONGA":longa,"N_TRUNCAMENTO":trunc,
            "N_ALONGAMENTO":along,"N_PREENCHIDA":preen,"N_PALAVRAS":pal,"DENS_HESIT_100":dens,
        })
        w.writerow(r)

# ---------- 5. checagem da hipotese: densidade de hesitacao por zona ----------
import statistics as st
out = list(csv.DictReader(open(OUT, encoding="utf-8")))
print("=== QTs por entrevista ===")
from collections import OrderedDict, defaultdict
seen=OrderedDict()
for r in out: seen[r["QT_ID"]]=r["ENTREVISTA"]
cnt=defaultdict(int)
for q,e in seen.items(): cnt[e]+=1
for e,c in cnt.items(): print(f"  {e}: {c} Quadros Topicos")

print("\n=== Densidade de hesitacao (marcas/100 palavras) por ZONA ===")
# foco em turnos do ENTREVISTADO com conteudo (exclui moldura/backchannel do entrevistador)
ENTREVISTADOS={"Marco Aurélio Mello","Galvão Bueno","Heloisa Starling"}
byz=defaultdict(list)
for r in out:
    if r["ZONA"] in ("fronteira","interior") and int(r["N_PALAVRAS"])>=8:
        byz[r["ZONA"]].append(float(r["DENS_HESIT_100"]))
for z in ("fronteira","interior"):
    v=byz[z]
    print(f"  {z:10s}: n={len(v):3d}  media={st.mean(v):.2f}  mediana={st.median(v):.2f}")

print("\n=== Apenas falas do ENTREVISTADO ===")
byz2=defaultdict(list)
for r in out:
    if r["LOCUTOR"] in ENTREVISTADOS and r["ZONA"] in ("fronteira","interior") and int(r["N_PALAVRAS"])>=8:
        byz2[r["ZONA"]].append(float(r["DENS_HESIT_100"]))
for z in ("fronteira","interior"):
    v=byz2[z]
    if v: print(f"  {z:10s}: n={len(v):3d}  media={st.mean(v):.2f}  mediana={st.median(v):.2f}")
print("\nOK ->", OUT)
