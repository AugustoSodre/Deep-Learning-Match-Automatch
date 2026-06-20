# 🧠 AutoMatch AI - Deep Learning Recommendation Engine

Este microsserviço é o "cérebro" do projeto **AutoMatch**. Ele utiliza técnicas avançadas de Deep Learning para calcular a afinidade entre o perfil de um usuário e as especificações técnicas de um veículo.

## 🔬 Arquitetura: Two-Tower Model

Implementamos uma **Arquitetura de Duas Torres (Two-Tower Architecture)** utilizando **PyTorch**. Esta arquitetura é ideal para sistemas de recomendação onde precisamos mapear entidades diferentes (Usuários e Itens) para um mesmo espaço vetorial (latent space).

- **User Tower (Torre do Usuário):** Codifica orçamento, prioridades (economia e potência) e categorias preferidas em um vetor latente.
- **Car Tower (Torre do Carro):** Codifica preço, potência, consumo e categoria do veículo em um vetor de mesma dimensão.
- **Scoring Híbrido:** O modelo calcula a similaridade de cosseno (`model_score`), que é depois combinado com um `preference_boost` baseado em uso principal, ambiente, grupo/família e faixa de ano do modelo. O resultado final é: `match_score = 0.38 * model_score + 0.62 * preference_boost`.

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

O serviço espera o perfil do usuário e uma lista de carros.

**Entradas esperadas:**
- `user_profile`: demographics, financials, technicalPreferences, priorities
- `available_cars[]`: id, nome, ano, preco, categoria, specs.potencia, specs.consumo

**Lógica de Match:**
1. Extrai números de specs (ex: "150 cv" → 150).
2. Normaliza valores e gera embeddings do usuário e carro.
3. Calcula similaridade de cosseno (two-tower).
4. Aplica preference_boost com pesos: uso (34%), categoria (28%), ambiente (22%), família (10%), ano (6%).
5. Combina scores com ponderação: 38% modelo + 62% preferência.
6. Retorna ranking ordenado por `match_score`.

---
Desenvolvido para o ecossistema **AutoMatch AI**.
