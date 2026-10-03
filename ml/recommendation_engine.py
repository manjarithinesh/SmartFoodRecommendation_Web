def generate_recommendation(
    diet_type,
    disease_type,
    cuisine,
    dietary_restriction,
    allergy
):

    # Default values
    recommended_foods = []
    foods_to_limit = []
    description = ""

    # =====================================
    # BALANCED DIET
    # =====================================

    if diet_type == "Balanced":

        description = (
            "A balanced diet focuses on a combination of "
            "carbohydrates, proteins, healthy fats, vegetables "
            "and fruits."
        )

        recommended_foods = [
            "Vegetables",
            "Fruits",
            "Whole grains",
            "Eggs",
            "Fish",
            "Lean chicken",
            "Dal and legumes",
            "Curd"
        ]

        foods_to_limit = [
            "Highly processed foods",
            "Sugary drinks",
            "Excess fried foods"
        ]


    # =====================================
    # LOW CARB DIET
    # =====================================

    elif diet_type == "Low_Carb":

        description = (
            "A low-carbohydrate diet focuses on protein-rich foods, "
            "vegetables and healthy fats while limiting high-carbohydrate foods."
        )

        recommended_foods = [
            "Eggs",
            "Fish",
            "Chicken",
            "Paneer",
            "Leafy vegetables",
            "Broccoli",
            "Cucumber",
            "Nuts"
        ]

        foods_to_limit = [
            "White rice",
            "Sugary foods",
            "Sweetened drinks",
            "Refined bread",
            "Large portions of potatoes"
        ]


    # =====================================
    # LOW SODIUM DIET
    # =====================================

    elif diet_type == "Low_Sodium":

        description = (
            "A low-sodium diet focuses on reducing foods with high "
            "salt content and choosing fresh, minimally processed foods."
        )

        recommended_foods = [
            "Fresh vegetables",
            "Fresh fruits",
            "Unsalted nuts",
            "Whole grains",
            "Fresh fish",
            "Beans",
            "Low-sodium foods"
        ]

        foods_to_limit = [
            "Pickles",
            "Chips",
            "Processed foods",
            "Instant noodles",
            "Processed meats",
            "Excess table salt"
        ]


    # =====================================
    # DISEASE-SPECIFIC ADDITIONS
    # =====================================

    if disease_type == "Diabetes":

        recommended_foods.extend([
            "Leafy vegetables",
            "Whole grains",
            "Legumes"
        ])

        foods_to_limit.extend([
            "Sugary drinks",
            "High-sugar foods"
        ])


    elif disease_type == "Hypertension":

        recommended_foods.extend([
            "Fresh vegetables",
            "Fruits",
            "Unsalted nuts"
        ])

        foods_to_limit.extend([
            "High-salt foods",
            "Processed foods"
        ])


    elif disease_type == "Obesity":

        recommended_foods.extend([
            "Vegetables",
            "Lean protein",
            "Whole grains"
        ])

        foods_to_limit.extend([
            "Fried foods",
            "Sugary drinks",
            "Highly processed foods"
        ])


    # =====================================
    # DIETARY RESTRICTIONS
    # =====================================

    if dietary_restriction == "Low_Sodium":

        foods_to_limit.extend([
            "High-sodium packaged foods"
        ])


    elif dietary_restriction == "Low_Sugar":

        foods_to_limit.extend([
            "Sweets",
            "Sugary beverages"
        ])


    # =====================================
    # ALLERGY
    # =====================================

    if allergy == "Gluten":

        foods_to_limit.extend([
            "Wheat-based foods"
        ])


    elif allergy == "Peanuts":

        foods_to_limit.extend([
            "Peanuts",
            "Foods containing peanuts"
        ])


    # Remove duplicate items
    recommended_foods = list(dict.fromkeys(recommended_foods))
    foods_to_limit = list(dict.fromkeys(foods_to_limit))


    return {
        "description": description,
        "recommended_foods": recommended_foods,
        "foods_to_limit": foods_to_limit
    }
