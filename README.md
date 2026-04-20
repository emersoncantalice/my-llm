# My LLM Studio

Projeto base para ter uma LLM propria com:
- Chat web
- Adicao de contexto (RAG)
- Fine-tuning com LoRA

## Stack
- `transformers` + `torch`
- `sentence-transformers` + `faiss-cpu` para contexto
- `peft` para fine-tuning eficiente
- `gradio` para interface

## Modelo padrao
- `sshleifer/tiny-gpt2` (leve, ideal para comecar)

Voce pode trocar o modelo com variavel de ambiente:

```powershell
$env:BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
```

## Estrutura

- `main.py`: inicializa interface web
- `src/my_llm/ui.py`: UI com abas de chat/contexto/fine-tuning
- `src/my_llm/context_store.py`: indexacao e recuperacao de contexto
- `src/my_llm/fine_tuner.py`: treino LoRA
- `data/contexts/`: documentos para contexto
- `data/finetune/train.jsonl`: dataset de fine-tuning

## Instalacao

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Rodar interface

```powershell
$env:PYTHONPATH="src"
python main.py
```

A interface sobe em `http://localhost:7860`.

## Adicionar contexto

Opcao 1: pela interface, aba **Add Context**.

Opcao 2: via script:

```powershell
$env:PYTHONPATH="src"
python scripts/ingest_context.py
```

Coloque arquivos `.txt`, `.md`, `.json` em `data/contexts/`.

## Fine-tuning

1. Preencha `data/finetune/train.jsonl` com JSONL no formato:

```json
{"instruction":"...","output":"..."}
```

2. Treine via interface na aba **Fine-Tuning** ou via script:

```powershell
$env:PYTHONPATH="src"
python scripts/train.py
```

Os adaptadores LoRA serao salvos em `data/finetune/adapters/`.

## Observacoes

- Para respostas melhores, troque o modelo padrao por um instruct pequeno.
- Em CPU, o treino pode ser lento.
- Este projeto e uma base inicial pronta para evoluir com autenticacao, historico persistente e avaliacao automatica.
