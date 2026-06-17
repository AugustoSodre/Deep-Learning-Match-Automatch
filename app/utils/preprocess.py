import re
import torch

def clean_numeric(text):
    """
    Extrai números de strings como '150 cv' ou '12.5 km/l'.
    """
    if isinstance(text, (int, float)):
        return float(text)
    
    match = re.search(r"(\d+\.?\d*)", str(text))
    return float(match.group(1)) if match else 0.0

def preprocess_user(user_data, category_to_idx):
    """
    Converte o JSON do usuário em tensores para a UserTower.
    """
    # Exemplo simplificado de mapeamento de prioridades (1-5) para tensor
    priorities = [
        user_data['priorities']['economy'],
        user_data['priorities']['power'],
        user_data['priorities']['comfort'],
        user_data['priorities']['safety']
    ]
    
    # Orçamento normalizado (exemplo: dividindo por 100k)
    budget = user_data['financials']['maxBudget'] / 100000.0
    
    # Categorias preferidas (Multi-hot encoding simplificado ou lista de índices)
    # Aqui usamos uma lista de índices para um EmbeddingBag ou similar
    cat_indices = [category_to_idx.get(c, 0) for c in user_data['technicalPreferences']['categories']]
    if not cat_indices:
        cat_indices = [0] # Índice para 'Desconhecido'
        
    return {
        "numeric_features": torch.tensor([budget] + priorities, dtype=torch.float32).unsqueeze(0),
        "category_features": torch.tensor(cat_indices, dtype=torch.long).unsqueeze(0)
    }

def preprocess_car(car_data, category_to_idx):
    """
    Converte o JSON do carro em tensores para a CarTower.
    """
    # Extração de specs
    hp = clean_numeric(car_data['specs'].get('potencia', 0))
    consumption = clean_numeric(car_data['specs'].get('consumo', 0))
    price = car_data.get('preco', 0) / 100000.0
    
    # Categoria única do carro
    cat_idx = category_to_idx.get(car_data.get('categoria'), 0)
    
    return {
        "numeric_features": torch.tensor([price, hp, consumption], dtype=torch.float32).unsqueeze(0),
        "category_features": torch.tensor([cat_idx], dtype=torch.long).unsqueeze(0)
    }
