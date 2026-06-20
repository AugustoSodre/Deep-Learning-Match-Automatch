import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, num_categories, embedding_dim, numeric_dim):
        super(UserTower, self).__init__()
        # EmbeddingBag é excelente para lidar com múltiplas categorias preferidas (média dos embeddings)
        self.embedding = nn.EmbeddingBag(num_categories, embedding_dim, mode='mean')
        
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + numeric_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32) # Espaço latente comum
        )
        
    def forward(self, numeric_f, category_f):
        # category_f: Tensor de índices de categorias
        embedded_cats = self.embedding(category_f)
        x = torch.cat([numeric_f, embedded_cats], dim=1)
        return self.net(x)

class CarTower(nn.Module):
    def __init__(self, num_categories, embedding_dim, numeric_dim):
        super(CarTower, self).__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + numeric_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32) # Deve ter o mesmo tamanho da UserTower
        )
        
    def forward(self, numeric_f, category_f):
        # category_f: Índice da categoria do carro
        embedded_cat = self.embedding(category_f).squeeze(1)
        x = torch.cat([numeric_f, embedded_cat], dim=1)
        return self.net(x)

class TwoTowerMatch(nn.Module):
    def __init__(self, num_categories, embedding_dim=16):
        super(TwoTowerMatch, self).__init__()
        # User: 3 features numéricas (budget + 2 prioridades)
        self.user_tower = UserTower(num_categories, embedding_dim, numeric_dim=3)
        # Car: 3 features numéricas (price + hp + consumption)
        self.car_tower = CarTower(num_categories, embedding_dim, numeric_dim=3)
        
    def forward(self, user_inputs, car_inputs):
        """
        Fluxo dos Tensores:
        1. UserTower processa perfil do usuário -> vetor 1x32
        2. CarTower processa specs do carro -> vetor 1x32
        3. Calcula-se a Similaridade de Cosseno entre ambos.
        """
        user_vector = self.user_tower(user_inputs['numeric_features'], user_inputs['category_features'])
        car_vector = self.car_tower(car_inputs['numeric_features'], car_inputs['category_features'])
        
        # Similaridade de Cosseno normalizada para [0, 1]
        # Mapeia -1 para 0%, 0 para 50% e 1 para 100% para melhor UX
        score = F.cosine_similarity(user_vector, car_vector)
        return (score + 1) / 2
