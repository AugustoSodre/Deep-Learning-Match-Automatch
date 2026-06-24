from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import torch

from app.models.model import TwoTowerMatch
from app.utils.preprocess import preprocess_user, preprocess_car
from app.utils.affinity import numeric_affinity

app = FastAPI(title="AutoMatch AI Engine", version="1.1.0")

CATEGORIES = ['Desconhecido', 'Hatch', 'Sedan', 'SUV', 'Picape', 'Eletrico', 'Premium', 'Popular']
CAT_TO_IDX = {cat: i for i, cat in enumerate(CATEGORIES)}

# Modelo nao-treinado gera saida ~aleatoria. Fixamos a semente para que o
# resultado seja ao menos REPRODUTIVEL, e mantemos o peso do modelo baixo
# (ele entra so como leve desempate). O fix definitivo e treinar a rede.
torch.manual_seed(42)
model = TwoTowerMatch(num_categories=len(CATEGORIES))
model.eval()

# Mapas de afinidade com CREDITO PARCIAL: (categorias_ideais, categorias_aceitaveis)
# ideal -> 1.0 | aceitavel -> 0.5 | nenhuma -> 0.15 (nunca zera de vez)
USE_CATEGORY_MAP = {
    "work_commute": ({"Hatch", "Sedan", "Eletrico", "Popular"}, {"SUV", "Premium"}),
    "travel":       ({"Sedan", "SUV", "Premium"}, {"Hatch", "Picape"}),
    "ride_hailing": ({"Hatch", "Sedan", "Eletrico"}, {"Popular", "Premium"}),
    "off_road":     ({"SUV", "Picape"}, {"Sedan"}),
}
ENV_CATEGORY_MAP = {
    "city":      ({"Hatch", "Sedan", "Eletrico", "Premium", "Popular"}, {"SUV"}),
    "highway":   ({"Sedan", "SUV", "Premium"}, {"Hatch", "Picape", "Eletrico"}),
    "dirt_road": ({"SUV", "Picape"}, {"Sedan"}),
}
FAMILY_CATEGORY_MAP = {
    "2":   ({"Hatch", "Sedan", "Eletrico", "Premium", "Popular"}, {"SUV"}),
    "3-4": ({"Sedan", "SUV", "Hatch"}, {"Picape", "Premium"}),
    "5+":  ({"SUV", "Picape"}, {"Sedan"}),
}
AGE_YEAR_MAP = {
    "0km": (2024, 2100),
    "up_to_3_years": (2022, 2026),
    "up_to_10_years": (2016, 2022),
}


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def soft_category_score(option_map: Dict[str, Any], key: str, car_category: str) -> float:
    """Credito parcial em vez de tudo-ou-nada."""
    pair = option_map.get(key)
    if not pair:
        return 0.5  # sem informacao -> neutro
    ideal, acceptable = pair
    if car_category in ideal:
        return 1.0
    if car_category in acceptable:
        return 0.5
    return 0.15


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
    demographics = user_data.get("demographics", {})
    tech = user_data.get("technicalPreferences", {})

    # Sinal mais forte: o usuario escolheu essa categoria explicitamente?
    selected_categories = set(tech.get("categories", []))
    explicit_score = 1.0 if car_category in selected_categories else 0.0

    # Sinais de contexto (presets) com credito parcial
    use_score = soft_category_score(USE_CATEGORY_MAP, demographics.get("primaryUse", ""), car_category)
    env_score = soft_category_score(ENV_CATEGORY_MAP, demographics.get("primaryEnvironment", ""), car_category)
    family_score = soft_category_score(FAMILY_CATEGORY_MAP, demographics.get("familySize", ""), car_category)
    year_score = score_year_fit(tech.get("vehicleAge", ""), car_data.get("ano"))

    context = (0.34 * use_score) + (0.22 * env_score) + (0.10 * family_score) + (0.06 * year_score)
    context = context / 0.72  # normaliza para [0, 1]

    # Categoria escolhida DOMINA; os presets sao apenas ajuste secundario.
    # Assim um carro que o usuario pediu nunca cai para ~45% por causa de presets.
    return clamp((0.55 * explicit_score) + (0.45 * context))


class MatchRequest(BaseModel):
    user_profile: Dict[str, Any]
    available_cars: List[Dict[str, Any]]


@app.post("/match")
async def get_matches(request: MatchRequest):
    """Calcula o match entre um usuario e uma lista de carros."""
    try:
        user_data = request.user_profile
        cars = request.available_cars
        user_tensors = preprocess_user(user_data, CAT_TO_IDX)
        results = []

        # Preco maximo e LIMITE RIGIDO: carro acima do teto e descartado
        # e nem chega a ser pontuado/exibido. (0 ou ausente = sem limite)
        max_budget = float(user_data.get("financials", {}).get("maxBudget", 0) or 0)

        with torch.no_grad():
            for car in cars:
                car_price = float(car.get("preco", 0) or 0)
                if max_budget > 0 and car_price > max_budget:
                    continue  # acima do orcamento -> fora da lista

                car_tensors = preprocess_car(car, CAT_TO_IDX)
                model_score = float(model(user_tensors, car_tensors).item())
                pref_score = preference_boost(user_data, car)
                num_score = numeric_affinity(user_data, car)

                # Blend: preferencia (categorica) + afinidade numerica dominam;
                # o modelo nao-treinado entra so como leve desempate (8%).
                final_score = clamp(
                    (0.50 * pref_score)
                    + (0.42 * num_score)
                    + (0.08 * model_score)
                )

                results.append({
                    "id": car.get("id"),
                    "nome": car.get("nome"),
                    "match_score": round(final_score, 4),
                    "preference_score": round(pref_score, 4),
                    "numeric_score": round(num_score, 4),
                    "model_score": round(model_score, 4),
                })

        results.sort(key=lambda x: x["match_score"], reverse=True)
        return {"status": "success", "matches": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
