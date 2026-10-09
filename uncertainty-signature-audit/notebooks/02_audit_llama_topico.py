# ==============================================================================
# Auditoria de Calibracao (ECE) - Llama 3.1 sobre o benchmark anotado por TOPICO
# Rode no Google Colab com GPU (T4). Gera 'final_audit_results.csv'.
# Antes de rodar: revogue o token antigo no Hugging Face e cole um TOKEN NOVO.
# ==============================================================================

# 1. INSTALACAO E SETUP
!pip install -q transformers accelerate bitsandbytes pandas huggingface_hub matplotlib seaborn

import torch, gc, sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

# VERIFICACAO DE GPU
if not torch.cuda.is_available():
    print("ERRO: GPU nao detectada! Ambiente de Execucao > Alterar tipo > T4 GPU.")
    sys.exit()

# 2. AUTENTICACAO  (cole aqui o TOKEN NOVO)
hf_token = "COLE_SEU_TOKEN_NOVO_AQUI"
login(token=hf_token)

model_id = "meta-llama/Llama-3.1-8B-Instruct"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

print("Carregando o modelo Llama 3.1... (leva alguns minutos)")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", quantization_config=quantization_config
)

# 3. CARREGAR O BENCHMARK ANOTADO (suba o arquivo na aba Files do Colab)
try:
    df = pd.read_csv("/content/stil_benchmark_topico_discursivo.csv")
    print(f"Sucesso: {len(df)} linhas para a auditoria.")
except FileNotFoundError:
    print("ERRO: 'stil_benchmark_topico_discursivo.csv' nao encontrado no Colab.")
    sys.exit()

# 4. FUNCAO DE CONFIANCA (com truncamento + limpeza de memoria)
def get_llm_confidence(text):
    prompt = f"Text: '{str(text)[:1500]}'. Does the speaker demonstrate absolute certainty? Answer only Yes or No.\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits[:, -1, :], dim=-1)
    id_yes = tokenizer.encode("Yes", add_special_tokens=False)[-1]
    conf = probs[0, id_yes].item() * 100
    del inputs, outputs, probs
    torch.cuda.empty_cache(); gc.collect()
    return conf

# 5. EXECUTAR A AUDITORIA
print(f"Iniciando auditoria probabilistica nos {len(df)} trechos...")
df['CONFIANCA_IA'] = df['FALA_HIGIENIZADA'].apply(get_llm_confidence)
df['ECE_ERROR'] = abs(df['CONFIANCA_IA'] - df['NOTA_HUMANA'])

# 6. ECE GLOBAL
ece_score = df['ECE_ERROR'].mean()
print(f"\n--- RESULTADO ---\nExpected Calibration Error (ECE) medio: {ece_score:.2f}%")

# 7. GRAFICO (Reliability Diagram)
plt.figure(figsize=(10, 6))
sns.regplot(x='NOTA_HUMANA', y='CONFIANCA_IA', data=df,
            scatter_kws={'alpha':0.4, 'color':'blue'},
            line_kws={'color':'red', 'label':'Tendencia da IA'})
plt.plot([0, 100], [0, 100], '--', color='gray', label='Calibracao Perfeita')
plt.title(f'Reliability Diagram: Llama 3.1 ({len(df)} trechos - ECE: {ece_score:.2f}%)')
plt.xlabel('Nota Humana (Ground Truth)'); plt.ylabel('Confianca da IA')
plt.legend(); plt.grid(True, alpha=0.3); plt.savefig("calibration_plot.png"); plt.show()

# 8. SALVAR
df.to_csv("final_audit_results.csv", index=False)
print("Gerados: 'final_audit_results.csv' e 'calibration_plot.png'")
