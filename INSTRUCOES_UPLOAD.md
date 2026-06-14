# Como subir para o GitHub

Tudo já está organizado nas pastas certas (`data/`, `src/`, `notebooks/`, `results/`, `docs/`),
espelhando a estrutura do seu repositório `uncertainty-signature-audit`.

**Antes de tudo:** confira que nenhum arquivo tem o seu token do Hugging Face.
O `notebooks/02_audit_llama_topico.py` usa só o placeholder `COLE_SEU_TOKEN_NOVO_AQUI` — está seguro.

---

## Opção A — Pelo site do GitHub (mais simples, sem terminal)

1. Abra https://github.com/ValeriaVSantos/uncertainty-signature-audit
2. Clique em **Add file → Upload files**.
3. Arraste as pastas `data`, `src`, `notebooks`, `results`, `docs` (de dentro de `github_upload`).
   O GitHub mantém a estrutura de pastas no arrasto.
4. Em "Commit changes", escreva a mensagem (ex.: *Add discourse-topic layer and calibration-by-topic analysis*).
5. **Commit**.
6. Cole o conteúdo de `README_SECAO_TOPICO.md` no final do seu `README.md` (edite o README pelo próprio site, botão de lápis).

> O `.gitignore` você também pode subir pelo mesmo upload (ele não aparece arrastando pastas; suba-o à parte ou crie pelo site com **Add file → Create new file**, nome `.gitignore`, e cole o conteúdo).

---

## Opção B — Pelo terminal (git), se preferir

```bash
# 1. clone o repositorio (se ainda nao tiver local)
git clone https://github.com/ValeriaVSantos/uncertainty-signature-audit.git
cd uncertainty-signature-audit

# 2. copie os arquivos preparados para dentro do repo
#    (ajuste o caminho de origem para onde esta a pasta github_upload)
cp -r /caminho/para/github_upload/data/*       data/
cp -r /caminho/para/github_upload/src/*        src/
cp -r /caminho/para/github_upload/notebooks/*  notebooks/
cp -r /caminho/para/github_upload/results/*    results/
cp -r /caminho/para/github_upload/docs/*       docs/
cp    /caminho/para/github_upload/.gitignore   .gitignore

# 3. atualize o README manualmente colando a secao de README_SECAO_TOPICO.md

# 4. commit e push
git add .
git commit -m "Add discourse-topic layer and calibration-by-topic analysis (Jubran 2015)"
git push origin main
```

---

## Decisão pendente: o ensaio é público?

A pasta `docs/ensaio/` contém o ensaio (docx + pdf). Se você pretende transformá-lo em artigo/submissão,
**não suba** essa pasta por enquanto (alguns periódicos implicam com texto já circulando) — basta não
arrastá-la na Opção A, ou apagá-la antes do commit na Opção B. Código e dados podem ir normalmente.
