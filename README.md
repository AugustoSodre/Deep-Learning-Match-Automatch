# Deep-Learning-Match-Automatch

### Como Iniciar o Projeto

Para colocar o motor de IA em funcionamento, siga estes passos:

1.  **Criar Ambiente Virtual (Recomendado):**
    ```bash
    cd Deep-Learning-Match-Automatch
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

2.  **Instalar Dependências:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Executar o Servidor:**
    ```bash
    python main.py
    ```
    O serviço estará rodando em `http://localhost:8000`. Você pode acessar a documentação interativa em `http://localhost:8000/docs`.

---

### README.md

Abaixo está o conteúdo do arquivo `README.md` estruturado para o repositório:

```markdown
# AutoMatch AI - Deep Learning Recommendation Engine

Este repositório contém o microsserviço de Inteligência Artificial para o projeto **AutoMatch**. Ele é responsável por realizar o "match" inteligente entre perfis de usuários e catálogos de veículos utilizando Deep Learning.

## 🧠 Arquitetura: Two-Tower Model

Utilizamos uma **Arquitetura de Duas Torres (Two-Tower Architecture)** implementada em **PyTorch**. Essa abordagem é amplamente utilizada por empresas como Netflix e YouTube para sistemas de recomendação em larga escala.

- **User Tower (Torre do Usuário):** Processa dados demográficos, financeiros e, principalmente, os pesos de prioridade (Economia, Potência, Conforto, Segurança) para gerar um embedding (vetor matemático) que representa o desejo do usuário.
- **Car Tower (Torre do Carro):** Processa especificações técnicas (HP, consumo, preço) e categorias do veículo para gerar um embedding no mesmo espaço latente da torre do usuário.
- **Similaridade de Cosseno:** O motor calcula a proximidade entre os dois vetores. Quanto mais próximos (maior o score), melhor é a recomendação para aquele usuário específico.

## 🛠️ Stack Tecnológica

- **Linguagem:** Python 3.9+
- **Framework Web:** FastAPI (Alta performance e validação automática com Pydantic)
- **Deep Learning:** PyTorch
- **Processamento de Dados:** RegEx e Tensores Torch
- **Servidor ASGI:** Uvicorn

## 📂 Estrutura do Projeto

```text
Deep-Learning-Match-Automatch/
├── app/
│   ├── api/            # Endpoints da API
│   ├── core/           # Configurações globais
│   ├── models/         # Definição das Redes Neurais (PyTorch)
│   │   └── model.py    # UserTower, CarTower e TwoTowerMatch
│   └── utils/          # Lógica de pré-processamento
│       └── preprocess.py # Limpeza de strings e normalização
├── main.py             # Ponto de entrada da aplicação
├── requirements.txt    # Dependências do sistema
└── README.md           # Documentação
```

## 🚀 Instalação e Execução

### Pré-requisitos
- Python 3.9 ou superior instalado.

### Passo a Passo

1. **Clonar o repositório:**
   ```bash
   git clone <url-do-repositorio>
   cd Deep-Learning-Match-Automatch
   ```

2. **Configurar ambiente virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Instalar dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Iniciar o serviço:**
   ```bash
   python main.py
   ```

## 🔌 API Endpoints

### `POST /match`
Recebe o perfil do usuário e a lista de carros disponíveis para calcular os scores de recomendação.

**Exemplo de Payload:**
```json
{
  "user_profile": {
    "financials": { "maxBudget": 120000 },
    "priorities": { "economy": 5, "power": 2, "comfort": 4, "safety": 5 },
    "technicalPreferences": { "categories": ["SUV", "Hatch"] }
  },
  "available_cars": [
    {
      "id": "1",
      "nome": "Carro X",
      "preco": 95000,
      "categoria": "SUV",
      "specs": { "potencia": "150 cv", "consumo": "12 km/l" }
    }
  ]
}
```

## 🔗 Integração com Frontend
O frontend Angular está configurado para enviar os dados coletados no Quiz de Onboarding diretamente para este microsserviço via porta `8000`. Certifique-se de que o CORS esteja configurado corretamente para o domínio do seu front.

---
Desenvolvido para o ecossistema **AutoMatch AI**.
```