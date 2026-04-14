#!/usr/bin/env python3
"""
Passo 1: Engenharia de Dados Sintéticos

Gera 55 pares (prompt, response) no domínio DevOps / Infraestrutura
utilizando a API OpenAI (GPT-3.5-turbo). Salva 90% como train.jsonl
e 10% como test.jsonl.

Uso:
    export OPENAI_API_KEY="sk-..."
    python 01_generate_dataset.py
"""

import os
import json
import random
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
DOMAIN = "DevOps e Infraestrutura de Software"
NUM_SAMPLES = 55          # >= 50 conforme requisito
TRAIN_RATIO = 0.9
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL = "gpt-3.5-turbo"   # Pode trocar por "gpt-4" para maior qualidade

# Tópicos para diversificar o dataset
TOPICS = [
    "Docker e containerização",
    "Kubernetes e orquestração de contêineres",
    "CI/CD com GitLab e Jenkins",
    "Infraestrutura como Código (Terraform, Ansible)",
    "Monitoramento com Prometheus e Grafana",
    "Configuração de redes e firewalls Linux",
    "Backup e restauração de bancos de dados PostgreSQL",
    "Git e controle de versão avançado",
    "Segurança em ambientes de nuvem",
    "Shell scripting e automação com Bash",
]

SYSTEM_PROMPT = (
    f"Você é um especialista sênior em {DOMAIN}. "
    "Gere um par de instrução e resposta técnica e detalhada. "
    "O par deve ser realista, didático e direto ao ponto. "
    "Responda APENAS em JSON com as chaves 'instruction' e 'response'."
)


def generate_pair(client: OpenAI, topic: str) -> dict | None:
    """Gera um único par (instruction, response) via OpenAI."""
    user_message = (
        f"Crie uma instrução técnica e sua resposta completa sobre o tópico: '{topic}'. "
        "A instrução deve ser uma pergunta ou tarefa prática do cotidiano de um DevOps Engineer. "
        "A resposta deve conter pelo menos 3 parágrafos ou passos detalhados."
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.8,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        pair = json.loads(content)
        # Valida campos obrigatórios
        if "instruction" in pair and "response" in pair:
            return {
                "instruction": pair["instruction"].strip(),
                "response": pair["response"].strip(),
            }
        return None
    except Exception as exc:
        print(f"  [WARN] Erro ao gerar par para '{topic}': {exc}")
        return None


def format_as_alpaca(pair: dict) -> dict:
    """Formata o par no estilo Alpaca (usado pelo SFTTrainer)."""
    return {
        "text": (
            f"### Instruction:\n{pair['instruction']}\n\n"
            f"### Response:\n{pair['response']}"
        )
    }


def save_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Salvo: {path} ({len(records)} registros)")


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Variável de ambiente OPENAI_API_KEY não definida. "
            "Execute: export OPENAI_API_KEY='sk-...'"
        )

    client = OpenAI(api_key=api_key)
    dataset: list[dict] = []

    print(f"Gerando {NUM_SAMPLES} pares de instrução/resposta no domínio: {DOMAIN}")
    print(f"Modelo OpenAI: {MODEL}\n")

    with tqdm(total=NUM_SAMPLES, desc="Gerando dataset") as pbar:
        topic_cycle = (TOPICS * (NUM_SAMPLES // len(TOPICS) + 1))[:NUM_SAMPLES]
        for topic in topic_cycle:
            pair = generate_pair(client, topic)
            if pair:
                dataset.append(format_as_alpaca(pair))
            pbar.update(1)

    print(f"\nPares gerados com sucesso: {len(dataset)}/{NUM_SAMPLES}")

    # Embaralha e divide
    random.seed(42)
    random.shuffle(dataset)
    split_idx = int(len(dataset) * TRAIN_RATIO)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]

    # Salva
    print("\nSalvando datasets...")
    save_jsonl(train_data, OUTPUT_DIR / "train.jsonl")
    save_jsonl(test_data, OUTPUT_DIR / "test.jsonl")

    print(f"\nDataset gerado com sucesso!")
    print(f"  Treino : {len(train_data)} amostras ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Teste  : {len(test_data)} amostras ({(1-TRAIN_RATIO)*100:.0f}%)")


if __name__ == "__main__":
    main()
