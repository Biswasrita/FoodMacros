import json
import os

def get_nutrition(food_label, mass_g):
    json_path = os.path.join(os.path.dirname(__file__), 'density_table.json')
    try:
        with open(json_path, 'r') as f:
            table = json.load(f)
    except Exception as e:
        print(f"Error loading density table: {e}")
        table = {}
    
    # Standardize label for lookup
    food_label = food_label.lower()
    
    if food_label in table:
        data = table[food_label]
    else:
        # Default fallback values for unrecognized foods
        data = {"density_g_cm3": 1.0, "calories": 100, "protein": 5, "fat": 5, "carbs": 10}
        
    multiplier = mass_g / 100.0
    return {
        "calories": data["calories"] * multiplier,
        "protein": data["protein"] * multiplier,
        "fat": data["fat"] * multiplier,
        "carbs": data["carbs"] * multiplier,
        "density_g_cm3": data["density_g_cm3"]
    }
