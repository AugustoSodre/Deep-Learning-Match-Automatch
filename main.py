from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import torch

from app.models.model import TwoTowerMatch
from app.utils.preprocess import preprocess_user, preprocess_car

app = FastAPI(title="AutoMatch AI Engine", version="1.0.0")

# Mapeamento fixo de categorias para exemplo
CATEGORIES = ['Desconhecido', 'Hatch', 'Sedan', 'SUV', 'Picape', 'Eletrico', 'Premium', 'Classico', 'Popular']
CAT_TO_IDX = {cat: i for i, cat in enumerate(CATEGORIES)}

# Inicializa o modelo (Em produção, carregaríamos pesos treinados)
model = TwoTowerMatch(num_categories=len(CATEGORIES))
model.eval()

class MatchRequest(BaseModel):
    user_profile: Dict[str, Any]
    available_cars: List[Dict[str, Any]]

@app.post("/match")
async def get_matches(request: MatchRequest):
    """
    Endpoint principal para calcular o match entre um usuário e uma lista de carros.
    Utiliza a arquitetura de Duas Torres para obter scores de afinidade.
    """
    try:
        user_data = request.user_profile
        cars = request.available_cars
        
        # 1. Pré-processamento do Usuário
        user_tensors = preprocess_user(user_data, CAT_TO_IDX)
        
        results = []
        
        # 2. Inferência (Batch processing seria ideal, aqui fazemos loop para clareza)
        with torch.no_grad():
            for car in cars:
                car_tensors = preprocess_car(car, CAT_TO_IDX)
                
                # Cálculo da similaridade
                score = model(user_tensors, car_tensors)
                
                results.append({
                    "id": car.get("id"),
                    "nome": car.get("nome"),
                    "match_score": float(score.item())
                })
        
        # Ordenar por maior score
        results.sort(key=lambda x: x["match_score"], reverse=True)
        
        return {
            "status": "success",
            "matches": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
