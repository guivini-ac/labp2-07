#!/usr/bin/env python3
"""
Passos 2-4: Fine-tuning QLoRA do Llama-2-7B

  Passo 2 – Quantização 4-bit (nf4) via bitsandbytes
  Passo 3 – Configuração LoRA: r=64, alpha=16, dropout=0.1
  Passo 4 – Treinamento com SFTTrainer:
              otimizador paged_adamw_32bit
              scheduler cosine
              warmup_ratio 0.03

Uso:
    python 02_finetune_qlora.py

Requisitos de hardware:
    GPU com >= 8 GB VRAM (ex: NVIDIA T4, A10G, A100).
    Recomenda-se executar no Google Colab (GPU runtime).
"""

import os
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer

# ---------------------------------------------------------------------------
# 0. Configurações gerais
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-2-7b-hf"   # Requer aceite dos termos na Hugging Face
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output/qlora-llama2-devops")
ADAPTER_DIR = Path("output/adapter")
MAX_SEQ_LENGTH = 512

print("=" * 60)
print("  Laboratório 07 – Fine-tuning QLoRA com LoRA")
print("=" * 60)

# ---------------------------------------------------------------------------
# Passo 2: Configuração da Quantização (QLoRA – 4-bit NF4)
# ---------------------------------------------------------------------------
print("\n[PASSO 2] Configurando quantização 4-bit (nf4)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat 4-bit
    bnb_4bit_compute_dtype=torch.float16, # Computação em float16
    bnb_4bit_use_double_quant=True,       # Double quantization para economia adicional
)

print("  BitsAndBytesConfig criado:")
print(f"    load_in_4bit         = True")
print(f"    bnb_4bit_quant_type  = 'nf4'")
print(f"    bnb_4bit_compute_dtype = float16")
print(f"    bnb_4bit_use_double_quant = True")

# ---------------------------------------------------------------------------
# Carregamento do modelo base com quantização
# ---------------------------------------------------------------------------
print(f"\nCarregando modelo base: {BASE_MODEL}")
print("  (Certifique-se de ter aceito os termos do Llama 2 na Hugging Face)")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",       # Distribui automaticamente entre GPU/CPU
    trust_remote_code=True,
)
model.config.use_cache = False                     # Desabilita cache para treino
model.config.pretraining_tp = 1                    # Evita divisão de tensores

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token          # Llama 2 não tem pad_token nativo
tokenizer.padding_side = "right"                   # Necessário para SFT estável

print("  Modelo carregado com sucesso!")

# ---------------------------------------------------------------------------
# Passo 3: Arquitetura LoRA
# ---------------------------------------------------------------------------
print("\n[PASSO 3] Configurando LoRA...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,    # Tarefa: Causal Language Modeling
    r=64,                             # Rank das matrizes de decomposição
    lora_alpha=16,                    # Fator de escala dos novos pesos
    lora_dropout=0.1,                 # Dropout para regularização
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Atenção
        "gate_proj", "up_proj", "down_proj",       # MLP (Feed-Forward)
    ],
)

print("  LoraConfig criado:")
print(f"    task_type    = CAUSAL_LM")
print(f"    r (rank)     = 64")
print(f"    lora_alpha   = 16")
print(f"    lora_dropout = 0.1")
print(f"    target_modules = q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj")

# Aplica LoRA ao modelo quantizado
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# Carregamento do Dataset
# ---------------------------------------------------------------------------
print("\nCarregando dataset...")

train_dataset = load_dataset(
    "json",
    data_files=str(DATA_DIR / "train.jsonl"),
    split="train",
)
test_dataset = load_dataset(
    "json",
    data_files=str(DATA_DIR / "test.jsonl"),
    split="train",
)

print(f"  Treino : {len(train_dataset)} amostras")
print(f"  Teste  : {len(test_dataset)} amostras")

# ---------------------------------------------------------------------------
# Passo 4: Pipeline de Treinamento
# ---------------------------------------------------------------------------
print("\n[PASSO 4] Configurando TrainingArguments e SFTTrainer...")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,          # Batch efetivo = 16
    gradient_checkpointing=True,            # Economiza VRAM ao custo de velocidade
    # -----------------------------------------------------------------------
    # Engenharia do Otimizador (requisito obrigatório)
    # -----------------------------------------------------------------------
    optim="paged_adamw_32bit",              # AdamW paginado: picos de memória → CPU
    learning_rate=2e-4,
    lr_scheduler_type="cosine",             # Taxa de aprendizado em curva cosseno
    warmup_ratio=0.03,                      # 3% do treino: warmup gradativo
    # -----------------------------------------------------------------------
    fp16=True,                              # Treino em meia precisão
    logging_steps=25,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",                       # Desabilita W&B / TensorBoard
    group_by_length=True,                   # Agrupa sequências de tamanho similar
)

print("  TrainingArguments configurados:")
print(f"    optim              = paged_adamw_32bit")
print(f"    lr_scheduler_type  = cosine")
print(f"    warmup_ratio       = 0.03")
print(f"    num_train_epochs   = 3")
print(f"    learning_rate      = 2e-4")
print(f"    fp16               = True")

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=lora_config,
    dataset_text_field="text",     # Coluna com o texto formatado no dataset
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,                 # Não agrupa múltiplas amostras numa sequência
)

# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------
print("\nIniciando treinamento...")
print("-" * 60)

trainer.train()

print("-" * 60)
print("\nTreinamento concluído!")

# ---------------------------------------------------------------------------
# Salvamento do Adaptador LoRA
# ---------------------------------------------------------------------------
print(f"\nSalvando adaptador LoRA em: {ADAPTER_DIR}")
trainer.model.save_pretrained(str(ADAPTER_DIR))    # Salva apenas os pesos LoRA
tokenizer.save_pretrained(str(ADAPTER_DIR))

print(f"  Adaptador salvo com sucesso em '{ADAPTER_DIR}'")
print("\n" + "=" * 60)
print("  Pipeline QLoRA finalizado!")
print("=" * 60)
