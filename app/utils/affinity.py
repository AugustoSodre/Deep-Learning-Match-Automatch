"""
Afinidade numerica do AutoMatch.

Antes, as features numericas (orcamento, potencia, consumo, prioridades)
entravam APENAS na rede neural nao-treinada -> viravam ruido.
Aqui calculamos um sinal numerico interpretavel e deterministico que
mede o quanto as specs do carro atendem ao que o usuario realmente prioriza.
"""
from .preprocess import clean_numeric


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def numeric_affinity(user_data: dict, car_data: dict) -> float:
    """Retorna um score em [0, 1] combinando ajuste de orcamento, economia e potencia."""
    budget = float(user_data.get("financials", {}).get("maxBudget", 0) or 0)
    price = float(car_data.get("preco", 0) or 0)

    # Orcamento: dentro do limite = 1.0; acima decai de forma suave (nao zera de imediato)
    if budget <= 0:
        budget_fit = 0.5
    elif price <= budget:
        budget_fit = 1.0
    else:
        budget_fit = clamp(1.0 - (price - budget) / budget)

    # Prioridades do usuario (1-5) normalizadas para [0, 1]
    economy_priority = clamp(user_data.get("priorities", {}).get("economy", 3) / 5.0)
    power_priority = clamp(user_data.get("priorities", {}).get("power", 3) / 5.0)

    # Specs do carro normalizadas: ~18 km/l = excelente consumo, ~300 cv = muito potente
    consumption = clean_numeric(car_data.get("specs", {}).get("consumo", 0))
    horsepower = clean_numeric(car_data.get("specs", {}).get("potencia", 0))
    economy_norm = clamp(consumption / 18.0)
    power_norm = clamp(horsepower / 300.0)

    # Alinhamento: quanto mais o carro entrega no que o usuario valoriza, maior o fit
    economy_fit = 1.0 - abs(economy_priority - economy_norm)
    power_fit = 1.0 - abs(power_priority - power_norm)

    return clamp(0.45 * budget_fit + 0.30 * economy_fit + 0.25 * power_fit)
