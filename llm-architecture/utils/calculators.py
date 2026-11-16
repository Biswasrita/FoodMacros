def calculate_bmr_tdee(weight_kg, height_cm, age, gender, activity_level):
    # Mifflin-St Jeor Equation
    if gender == "Male":
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161

    multipliers = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725
    }
    tdee = bmr * multipliers.get(activity_level, 1.2)
    return bmr, tdee