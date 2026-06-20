from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import torch

from app.models.model import TwoTowerMatch
from app.utils.preprocess import preprocess_user, preprocess_car

app = FastAPI(title="AutoMatch AI Engine", version="1.0.0")

# Mapeamento fixo de categorias para exemplo
CATEGORIES = ['Desconhecido', 'Hatch', 'Sedan', 'SUV', 'Picape', 'Eletrico', 'Premium', 'Popular']
CAT_TO_IDX = {cat: i for i, cat in enumerate(CATEGORIES)}

# Inicializa o modelo (Em produção, carregaríamos pesos treinados)
model = TwoTowerMatch(num_categories=len(CATEGORIES))
model.eval()

USE_CATEGORY_MAP = {
    "work_commute": {"Hatch", "Sedan", "Eletrico", "Premium"},
    "travel": {"Sedan", "SUV", "Premium"},
    "ride_hailing": {"Hatch", "Sedan", "Eletrico"},
    "off_road": {"SUV", "Picape"},
}

ENV_CATEGORY_MAP = {
    "city": {"Hatch", "Sedan", "Eletrico", "Premium"},
    "highway": {"Sedan", "SUV", "Premium"},
    "dirt_road": {"SUV", "Picape"},
}

FAMILY_CATEGORY_MAP = {
    "2": {"Hatch", "Sedan", "Eletrico", "Premium"},
    "3-4": {"Sedan", "SUV", "Picape"},
    "5+": {"SUV", "Picape"},
}

AGE_YEAR_MAP = {
    "0km": (2024, 2100),
    "up_to_3_years": (2022, 2026),
    "up_to_10_years": (2016, 2022),
}


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def score_option_match(allowed_categories: set[str], car_category: str) -> float:
    return 1.0 if car_category in allowed_categories else 0.0


def score_year_fit(vehicle_age: str, car_year: int | None) -> float:
    if not car_year:
        return 0.5

    year_range = AGE_YEAR_MAP.get(vehicle_age)
    if not year_range:
        return 0.5

    min_year, max_year = year_range
    if min_year <= car_year <= max_year:
        return 1.0

    distance = min(abs(car_year - min_year), abs(car_year - max_year))
    return clamp(1.0 - (distance / 10.0))


def preference_boost(user_data: Dict[str, Any], car_data: Dict[str, Any]) -> float:
    car_category = str(car_data.get("categoria", "Desconhecido"))

    selected_categories = set(user_data.get("technicalPreferences", {}).get("categories", []))
    selected_category_score = 1.0 if car_category in selected_categories else 0.0

    use_score = score_option_match(
        USE_CATEGORY_MAP.get(user_data.get("demographics", {}).get("primaryUse", ""), set()),
        car_category,
    )
    environment_score = score_option_match(
        ENV_CATEGORY_MAP.get(user_data.get("demographics", {}).get("primaryEnvironment", ""), set()),
        car_category,
    )
    family_score = score_option_match(
        FAMILY_CATEGORY_MAP.get(user_data.get("demographics", {}).get("familySize", ""), set()),
        car_category,
    )
    year_score = score_year_fit(
        user_data.get("technicalPreferences", {}).get("vehicleAge", ""),
        car_data.get("ano"),
    )

    return clamp(
        (0.28 * selected_category_score)
        + (0.34 * use_score)
        + (0.22 * environment_score)
        + (0.10 * family_score)
        + (0.06 * year_score)
    )

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
                score = float(model(user_tensors, car_tensors).item())
                preference_score = preference_boost(user_data, car)
                final_score = clamp((0.38 * score) + (0.62 * preference_score))
                
                results.append({
                    "id": car.get("id"),
                    "nome": car.get("nome"),
                    "match_score": final_score,
                    "model_score": score,
                    "preference_score": preference_score
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
