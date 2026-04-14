# Laboratório 07: Especialização de LLMs com LoRA e QLoRA

## Objetivo

Construir um pipeline completo de fine-tuning de um modelo de linguagem fundacional (Llama 2 7B) utilizando técnicas de eficiência de parâmetros (PEFT/LoRA) e quantização (QLoRA) para viabilizar o treinamento em hardwares limitados.

## Estrutura do Projeto

```
.
├── README.md
├── requirements.txt
├── 01_generate_dataset.py     # Passo 1: Geração do dataset sintético
├── 02_finetune_qlora.py       # Passos 2-4: Treinamento QLoRA completo
└── data/
    ├── train.jsonl            # Dataset de treino (90%)
    └── test.jsonl             # Dataset de teste (10%)
```

## Passos Implementados

### Passo 1 – Engenharia de Dados Sintéticos
- Script `01_generate_dataset.py` gera 50+ pares prompt/resposta via API OpenAI (GPT-3.5/GPT-4).
- Domínio escolhido: **DevOps / Infraestrutura de Software**.
- Divisão: 90% treino, 10% teste — salvo em `.jsonl`.

### Passo 2 – Configuração da Quantização (QLoRA)
- Uso da biblioteca `bitsandbytes`.
- `BitsAndBytesConfig` com `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=float16`.

### Passo 3 – Arquitetura LoRA
- Uso da biblioteca `peft`.
- `LoraConfig` com `task_type=CAUSAL_LM`.
- `r=64`, `lora_alpha=16`, `lora_dropout=0.1`.

### Passo 4 – Pipeline de Treinamento
- `SFTTrainer` da biblioteca `trl`.
- Otimizador: `paged_adamw_32bit`.
- Scheduler: `cosine`.
- Warmup ratio: `0.03`.
- Adaptador salvo via `trainer.model.save_pretrained`.

## Como Executar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Gerar o dataset
```bash
export OPENAI_API_KEY="sua-chave-aqui"
python 01_generate_dataset.py
```

### 3. Executar o treinamento
```bash
python 02_finetune_qlora.py
```

> **Nota:** O treinamento requer GPU com pelo menos 8GB de VRAM (ex: NVIDIA T4/A10G). Recomenda-se executar no Google Colab (GPU runtime) ou em instância com GPU.

## Versão

Entrega avaliativa marcada com a tag `v1.0`.
