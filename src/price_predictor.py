import numpy as np
import pandas as pd

# Base crop prices (Rs per quintal - approximate 2026 prices)
CROP_BASE_PRICES = {
    'Rice':         {'min': 2183, 'max': 3500,  'msp': 2183},
    'Wheat':        {'min': 2275, 'max': 3200,  'msp': 2275},
    'Sugarcane':    {'min': 340,  'max': 450,   'msp': 340},
    'Cotton':       {'min': 6620, 'max': 8500,  'msp': 6620},
    'Maize':        {'min': 1962, 'max': 2800,  'msp': 1962},
    'Bajra':        {'min': 2500, 'max': 3500,  'msp': 2500},
    'Jowar':        {'min': 3180, 'max': 4200,  'msp': 3180},
    'Moong':        {'min': 8558, 'max': 10000, 'msp': 8558},
    'Moth Bean':    {'min': 6300, 'max': 8000,  'msp': 6300},
    'Cluster Bean': {'min': 4500, 'max': 6000,  'msp': 4500},
    'Soybean':      {'min': 4600, 'max': 6000,  'msp': 4600},
    'Groundnut':    {'min': 6377, 'max': 8000,  'msp': 6377},
    'Sunflower':    {'min': 6760, 'max': 8500,  'msp': 6760},
    'Pulses':       {'min': 6000, 'max': 8000,  'msp': 6000},
    'Amaranth':     {'min': 3000, 'max': 5000,  'msp': 3000},
}

CROP_YIELD = {
    'Rice':         {'yield_qtl': 25, 'cost_per_acre': 15000},
    'Wheat':        {'yield_qtl': 20, 'cost_per_acre': 12000},
    'Sugarcane':    {'yield_qtl': 300,'cost_per_acre': 25000},
    'Cotton':       {'yield_qtl': 8,  'cost_per_acre': 18000},
    'Maize':        {'yield_qtl': 22, 'cost_per_acre': 10000},
    'Bajra':        {'yield_qtl': 12, 'cost_per_acre': 6000},
    'Jowar':        {'yield_qtl': 10, 'cost_per_acre': 5500},
    'Moong':        {'yield_qtl': 5,  'cost_per_acre': 8000},
    'Moth Bean':    {'yield_qtl': 4,  'cost_per_acre': 5000},
    'Cluster Bean': {'yield_qtl': 6,  'cost_per_acre': 6000},
    'Soybean':      {'yield_qtl': 8,  'cost_per_acre': 9000},
    'Groundnut':    {'yield_qtl': 10, 'cost_per_acre': 12000},
    'Sunflower':    {'yield_qtl': 8,  'cost_per_acre': 8000},
    'Pulses':       {'yield_qtl': 6,  'cost_per_acre': 7000},
    'Amaranth':     {'yield_qtl': 8,  'cost_per_acre': 4000},
}

def predict_crop_price(crop_name, drought_level, acres=1):
    if crop_name not in CROP_BASE_PRICES:
        return None

    price_info = CROP_BASE_PRICES[crop_name]
    yield_info = CROP_YIELD.get(crop_name,
                    {'yield_qtl': 10, 'cost_per_acre': 8000})

    # Drought impact on price
    drought_price_factor = {
        'normal':   1.0,
        'mild':     1.1,
        'moderate': 1.25,
        'severe':   1.45,
    }
    factor = drought_price_factor.get(drought_level, 1.0)

    # Drought impact on yield
    drought_yield_factor = {
        'normal':   1.0,
        'mild':     0.85,
        'moderate': 0.65,
        'severe':   0.40,
    }
    yield_factor = drought_yield_factor.get(drought_level, 1.0)

    # Calculations
    predicted_price = price_info['min'] * factor
    predicted_yield = yield_info['yield_qtl'] * yield_factor * acres
    total_revenue   = predicted_price * predicted_yield
    total_cost      = yield_info['cost_per_acre'] * acres
    profit          = total_revenue - total_cost
    roi             = (profit / total_cost * 100) if total_cost > 0 else 0

    return {
        'crop':            crop_name,
        'predicted_price': round(predicted_price, 0),
        'msp':             price_info['msp'],
        'min_price':       price_info['min'],
        'max_price':       price_info['max'],
        'predicted_yield': round(predicted_yield, 1),
        'total_revenue':   round(total_revenue, 0),
        'total_cost':      round(total_cost, 0),
        'profit':          round(profit, 0),
        'roi':             round(roi, 1),
        'acres':           acres,
    }

def get_best_crop_by_profit(crops_list, drought_level, acres=1):
    results = []
    for crop in crops_list:
        result = predict_crop_price(crop, drought_level, acres)
        if result:
            results.append(result)
    results.sort(key=lambda x: x['profit'], reverse=True)
    return results