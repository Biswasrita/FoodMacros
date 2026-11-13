
import os
import json
import re
import math

from google import genai
from PIL import Image


class LLMFoodClassifier:

    def __init__(self, api_key=None):

        key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()

        if not key:
            raise ValueError("Gemini API Key is missing.")

        self.client = genai.Client(api_key=key)

        # Primary and fallback models
        self.models = [
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash"
        ]

    def predict(self, crop_np):

        if crop_np is None or crop_np.size == 0:
            return self._default_response()

        image = Image.fromarray(crop_np).convert("RGB")

        prompt = """
        You are an expert food recognition and nutritional estimation AI specializing in global cuisines, including Indian, Bengali, Mexican, Italian, Chinese, Japanese, Korean, Mediterranean, American, Middle Eastern, and Southeast Asian cuisines.

        Analyze the provided cropped food image using a hierarchical, decision-tree-based visual recognition approach.
        
        Follow these steps internally before generating the final output.
        
        ### STEP 1: Identify the food category and cuisine
        
        First, determine the broad food category:
        
        - Main course
        - Rice-based dish
        - Bread-based dish
        - Curry or gravy
        - Fried food
        - Snack or street food
        - Dessert or confectionery
        - Salad
        - Soup
        - Beverage
        - Meat or seafood preparation
        - Dairy-based food
        - Fruit or vegetable preparation
        
        Next, identify the most likely cuisine based on visual characteristics, ingredients, preparation methods, shape, texture, garnishing, and presentation.
        
        Consider regional variations instead of treating all foods within a country as identical.
        
        For example:
        
        **Indian cuisine:**
        - Bengali: Macher Jhol, Kosha Mangsho, Shukto, Luchi, Aloo Posto, Mishti Doi, Shorshe Ilish, Cholar Dal, Bhapa Pitha.
        - North Indian: Butter Chicken, Dal Makhani, Chole Bhature, Rajma Chawal, Paneer Tikka.
        - South Indian: Dosa, Idli, Sambar, Uttapam, Appam, Puttu.
        - Street food: Pani Puri, Samosa, Kachori, Pav Bhaji, Vada Pav.
        
        **Mexican cuisine:**
        - Tacos
        - Burritos
        - Quesadillas
        - Enchiladas
        - Nachos
        - Tostadas
        - Tamales
        - Chilaquiles
        - Mexican rice
        - Guacamole
        
        Distinguish between tacos, burritos, and quesadillas using their shape, tortilla folding, filling, and cooking method.
        
        **Italian cuisine:**
        - Pizza
        - Pasta
        - Lasagna
        - Risotto
        - Ravioli
        - Gnocchi
        - Bruschetta
        - Focaccia
        
        **Chinese cuisine:**
        - Fried rice
        - Chow mein
        - Dumplings
        - Spring rolls
        - Manchurian
        - Sweet and sour preparations
        - Dim sum
        
        **Japanese cuisine:**
        - Sushi
        - Ramen
        - Tempura
        - Onigiri
        - Takoyaki
        - Udon
        - Bento preparations
        
        **Other cuisines:**
        Recognize Middle Eastern, Thai, Korean, Mediterranean, American, and other regional preparations. Do not force a dish into one of the examples above.
        
        ### STEP 2: Identify the specific food item
        
        Use the following decision hierarchy:
        
        1. Identify the dominant visible ingredients.
        2. Examine the shape, structure, and arrangement of the food.
        3. Identify cooking indicators such as frying, grilling, roasting, steaming, boiling, or baking.
        4. Examine sauces, gravies, spices, garnishes, and accompanying ingredients.
        5. Compare these visual characteristics with dishes from the identified cuisine.
        6. Select the most specific food identification supported by the image.
        
        Avoid overly generic labels such as "rice", "bread", "meat", or "dessert" when a more specific identification is visually possible.
        
        For example:
        - Prefer "Bengali fish curry" over "curry".
        - Prefer "Mexican chicken burrito" over "wrap".
        - Prefer "Paneer butter masala" over "paneer dish".
        - Prefer "Chicken biryani" over "rice".
        - Prefer "Chocolate lava cake" over "cake".
        
        However, never invent ingredients or assume a specific regional preparation without sufficient visual evidence.
        
        If multiple dishes look similar, select the most probable identification and reduce confidence accordingly.
        
        ### STEP 3: Estimate the portion size
        
        Estimate the visible serving size rather than assuming a standard serving.
        
        Determine the portion unit based on the food's physical structure:
        
        - 1 bowl
        - 1 plate
        - 1 cup
        - 1 slice
        - 1 piece
        - 1 scoop
        - 1 roll
        - 1 wrap
        - 1 taco
        - 1 serving
        - 1 glass
        - 1 ladle
        - 1 skewer
        
        Consider:
        - Visible dimensions and food volume.
        - Number of individual pieces.
        - Container or plate size, if visible.
        - Thickness and density.
        - Amount of filling, sauce, gravy, and toppings.
        
        Do not confuse the container size with the actual food quantity.
        
        For example, a bowl containing half a bowl of rice should not be estimated as a full bowl.
        
        ### STEP 4: Estimate macronutrients
        
        Estimate nutritional values for the entire visible portion, not per 100 grams.
        
        Use food-specific nutritional reasoning.
        
        Consider:
        - Main ingredients and their approximate proportions.
        - Cooking oil, butter, ghee, cheese, cream, and sauces.
        - Protein sources such as meat, fish, eggs, paneer, tofu, legumes, and dairy.
        - Carbohydrate sources such as rice, flour, tortillas, potatoes, and sugar.
        - Fiber from vegetables, legumes, fruits, and whole grains.
        - Regional preparation differences.
        
        Cuisine-specific considerations:
        
        **Bengali food:**
        Account for mustard oil, fish, rice, potatoes, poppy seed paste, coconut, and traditional sweets where visually applicable.
        
        **Mexican food:**
        Account for tortilla type, rice, beans, meat, cheese, sour cream, guacamole, and cooking oil.
        
        **Italian food:**
        Account for pasta or dough quantity, cheese, olive oil, cream, meat, and sauces.
        
        **Chinese food:**
        Account for stir-frying oil, rice or noodles, batter, sauces, meat, and vegetables.
        
        **Indian food:**
        Account for ghee, cooking oil, cream, paneer, rice, flour, lentils, and gravies.
        
        Do not assign identical nutritional values to visually similar foods prepared using different ingredients or cooking methods.
        
        Avoid false precision. Use reasonable estimates based on the visible portion and typical recipes.
        
        ### STEP 5: Confidence assessment
        
        Assign a confidence score between 0 and 1 based on:
        - Clarity of food identification.
        - Visibility of ingredients.
        - Distinctiveness of the preparation.
        - Portion estimation certainty.
        
        Use lower confidence when:
        - The image is heavily cropped.
        - Multiple dishes have similar appearances.
        - Ingredients are hidden beneath sauces or gravies.
        - The portion size cannot be reliably determined.
        
        ### STEP 6: Final output
        
        Return ONLY one valid JSON object using exactly the following schema.
        
        Do not add additional keys.
        Do not include explanations, assumptions, comments, or markdown.
        All numerical nutritional values must be non-negative.
        Calories must be an integer.
        Protein, fat, carbohydrates, and fiber must be numeric values in grams.
        
        {
            "label": "chocolate cake",
            "portion_unit": "1 slice",
            "confidence": 0.95,
            "calories": 350,
            "protein_g": 5.0,
            "fat_g": 14.0,
            "carbs_g": 48.0,
            "fiber_g": 2.5
        }
        """

        last_error = None

        for index, model in enumerate(self.models):

            try:
                print(f"Trying Gemini model: {model}")

                response = self.client.models.generate_content(
                    model=model,
                    contents=[prompt, image]
                )

            except Exception as error:

                last_error = error

                # Only use fallback for suitable API errors
                if index == 0 and self._should_fallback(error):

                    print(
                        f"Primary model failed: {error}. "
                        f"Trying {self.models[1]}..."
                    )
                    continue

                raise RuntimeError(
                    f"Gemini API failed ({model}): {error}"
                ) from error

            try:
                response_text = response.text or ""

                data = self._parse_json(response_text)

                result = self._format_response(data)

                print(f"Gemini succeeded: {model}")

                return result

            except (json.JSONDecodeError, ValueError, TypeError) as error:

                last_error = error

                if index == 0:
                    print(
                        f"Invalid response from {model}: {error}. "
                        f"Trying fallback..."
                    )
                    continue

                raise ValueError(
                    f"Gemini returned invalid data: {error}"
                ) from error

        raise RuntimeError(
            f"All Gemini models failed. Last error: {last_error}"
        )

    # Decide whether fallback is appropriate
    def _should_fallback(self, error):

        status = getattr(error, "status_code", None)

        if status is None:
            status = getattr(error, "code", None)

        try:
            status = int(status)
        except (TypeError, ValueError):
            status = None

        # Retry another model for these API failures
        if status in {404, 408, 429, 500, 502, 503, 504}:
            return True

        message = str(error).upper()

        return any(term in message for term in [
            "NOT_FOUND",
            "RESOURCE_EXHAUSTED",
            "UNAVAILABLE",
            "DEADLINE_EXCEEDED"
        ])

    # Extract JSON
    def _parse_json(self, response_text):

        text = response_text.strip()

        text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            text,
            flags=re.IGNORECASE
        ).strip()

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("No JSON found in Gemini response.")

        return json.loads(text[start:end + 1])

    # Validate and format output
    def _format_response(self, data):

        label = str(data.get("label", "unknown")).strip().lower()

        if not label or label == "unknown":
            raise ValueError("Food label is missing.")

        def safe_float(value):

            number = float(value)

            if not math.isfinite(number) or number < 0:
                raise ValueError("Invalid nutritional value.")

            return number

        return {
            "label": label,
            "portion_unit": str(
                data.get("portion_unit", "Unknown")
            ),
            "confidence": safe_float(data.get("confidence", 0)),
            "calories": safe_float(data.get("calories", 0)),
            "protein": safe_float(data.get("protein_g", 0)),
            "fat": safe_float(data.get("fat_g", 0)),
            "carbs": safe_float(data.get("carbs_g", 0)),
            "fiber": safe_float(data.get("fiber_g", 0))
        }

    # Default response for empty images
    def _default_response(self):

        return {
            "label": "unknown",
            "portion_unit": "Unknown",
            "confidence": 0.0,
            "calories": 0.0,
            "protein": 0.0,
            "fat": 0.0,
            "carbs": 0.0,
            "fiber": 0.0
        }
