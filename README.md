# 🧠 AutoMatch AI - Deep Learning Recommendation Engine

Este microsserviço é o "cérebro" do projeto **AutoMatch**. Ele utiliza técnicas avançadas de Deep Learning para calcular a afinidade entre o perfil de um usuário e as especificações técnicas de um veículo.

## 🔬 Arquitetura: Two-Tower Model

Implementamos uma **Arquitetura de Duas Torres (Two-Tower Architecture)** utilizando **PyTorch**. Esta arquitetura é ideal para sistemas de recomendação onde precisamos mapear entidades diferentes (Usuários e Itens) para um mesmo espaço vetorial (latent space).

- **User Tower (Torre do Usuário):** Codifica as preferências coletadas no Quiz (Orçamento, Prioridades de IA, Uso principal) em um vetor numérico (embedding).
- **Car Tower (Torre do Carro):** Codifica os atributos técnicos do veículo (Potência, Consumo, Categoria, Preço) em um vetor de mesma dimensão.
- **Normalização & Score:** Utilizamos a **Similaridade de Cosseno** normalizada para o intervalo `[0, 1]` para determinar o "Match Percentage".

## 🛠️ Stack Tecnológica

- **Linguagem:** Python 3.9+
- **Framework Web:** [FastAPI](https://fastapi.tiangolo.com/) (Alta performance e tipagem com Pydantic)
- **Deep Learning:** [PyTorch](https://pytorch.org/)
- **Processamento de Dados:** RegEx para extração de specs e Normalização linear.
- **Servidor:** Uvicorn

## 📂 Organização do Projeto

```text
Deep-Learning-Match-Automatch/
├── app/
│   ├── api/            # Definição dos endpoints REST
│   ├── models/         # Redes Neurais PyTorch (UserTower, CarTower)
│   └── utils/          # Lógica de extração de dados (ex: "150 cv" -> 150)
├── main.py             # Ponto de entrada (FastAPI)
├── requirements.txt    # Dependências do projeto
└── README.md           # Você está aqui
```

## 🚀 Como Iniciar

### Passo 1: Ambiente Virtual
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

### Passo 2: Dependências
```bash
pip install -r requirements.txt
```

### Passo 3: Execução
```bash
python main.py
```
O serviço estará disponível em `http://localhost:8000`. Acesse `/docs` para a documentação interativa.

## 🔌 API Endpoint: `POST /match`

O serviço espera um objeto contendo o perfil do usuário e uma lista de carros.

**Lógica de Match:**
O motor processa strings como `"12 km/l"` ou `"150 cv"`, normaliza os valores e aplica os pesos de prioridade do usuário (Economia, Potência, etc.) para gerar o ranking final.

---
Desenvolvido para o ecossistema **AutoMatch AI**.
