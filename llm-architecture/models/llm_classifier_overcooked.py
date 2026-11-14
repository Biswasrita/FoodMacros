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
        You are an expert food recognition, nutritional estimation,
        and visual food-condition assessment AI.

        Analyze the provided cropped food image carefully.

        STEP 1: FOOD IDENTIFICATION

        Identify:
        - The specific food item or dish.
        - Its likely cuisine and food category.
        - The dominant visible ingredients.
        - The cooking method, if visually identifiable.

        Recognize international cuisines, including Indian, Bengali,
        Mexican, Italian, Chinese, Japanese, Korean, Mediterranean,
        American, Middle Eastern and Southeast Asian.

        Prefer specific food names over generic labels.

        Examples:
        - Bengali fish curry instead of curry.
        - Chicken biryani instead of rice.
        - Paneer butter masala instead of paneer dish.
        - Chocolate lava cake instead of cake.

        Do not invent ingredients or regional variations.
        Reduce confidence when identification is uncertain.

        STEP 2: PORTION ESTIMATION

        Estimate the quantity of the visible food.

        Possible units:
        - 1 bowl
        - 1 plate
        - 1 cup
        - 1 slice
        - 1 piece
        - 1 serving
        - 1 roll
        - 1 wrap
        - 1 glass
        - 1 skewer

        Consider visible dimensions, food volume, number of pieces,
        container size and quantity of ingredients.

        Estimate the actual food portion, not the container capacity.

        Also estimate the mass of the visible edible food portion in grams.
        - Exclude plates, bowls, packaging and other containers.
        - Consider the visible quantity, number and size of pieces,
          food type and approximate density.
        - Use any visible size reference only when it is genuinely useful.
        - If there is not enough visual information to estimate weight,
          return 0 for estimated_weight_g and 0 for weight_confidence.
        - Otherwise return a reasonable approximate whole-number weight
          in grams and a weight_confidence between 0 and 1.
        - Do not imply this is a measured or calibrated weight.

        Nutritional estimates must refer to the same visible portion
        used for estimated_weight_g.

        STEP 3: FOOD CONDITION ASSESSMENT

        Carefully examine the visible condition of the food.

        Determine whether the food shows signs of:

        A. NORMAL APPEARANCE
        - Normal colour and appearance for the identified food.
        - No obvious visible signs of burning or spoilage.

        B. OVERCOOKED OR BURNT
        Look for:
        - Excessively blackened or charred surfaces.
        - Burnt edges or extensively darkened areas.
        - Visible signs of excessive heat exposure.
        - Unusually dried or heavily hardened surfaces
          when these are visually apparent.

        Distinguish actual burning from normal browning,
        grilling, roasting, caramelization, dark sauces and spices.

        Do not classify naturally dark foods as burnt.

        C. POSSIBLE SPOILAGE SIGNS

        Look for visible indicators such as:
        - Fuzzy green, white or black mold.
        - Unusual discoloration inconsistent with the food.
        - Visible signs of decay or deterioration.
        - Clearly abnormal surface changes.

        Do not assume that ordinary sauces, seasoning,
        natural food colours or condensation indicate spoilage.

        D. UNCERTAIN

        Use this category when the image does not provide
        enough visual evidence to assess the food condition.

        Return exactly one of these condition labels:

        - normal_appearance
        - overcooked_or_burnt
        - possible_spoilage_signs
        - uncertain

        Also assess condition severity:

        - none
        - mild
        - moderate
        - severe
        - unknown

        Provide a short description of the visible evidence.

        Assign a condition confidence between 0 and 1.

        IMPORTANT:
        Only report what is visually observable.
        Do not invent smells, tastes, storage history or expiry dates.
        Do not claim food is safe or unsafe based only on its appearance.

        STEP 4: NUTRITIONAL ESTIMATION

        Estimate nutritional values for the entire visible portion.

        Consider:
        - Main ingredients and their approximate quantities.
        - Cooking oil, butter, ghee, cheese and cream.
        - Protein sources such as meat, fish, eggs, paneer and legumes.
        - Carbohydrate sources such as rice, flour, potatoes and sugar.
        - Fibre from vegetables, fruits and whole grains.
        - Sauces, gravies, toppings and cooking methods.

        Account for cuisine-specific preparation differences.

        For example:
        - Bengali food: mustard oil, fish, rice and poppy seed paste.
        - Indian food: ghee, cream, paneer, rice and lentils.
        - Mexican food: tortillas, beans, meat and cheese.
        - Italian food: pasta, cheese, olive oil and sauces.
        - Chinese food: stir-frying oil, rice, noodles and sauces.

        Estimate:
        - Calories in kcal.
        - Protein in grams.
        - Fat in grams.
        - Carbohydrates in grams.
        - Fibre in grams.

        Estimate values for the complete visible portion,
        not per 100 grams.

        Avoid false precision.

        STEP 5: CONFIDENCE ASSESSMENT

        Food identification confidence should reflect:
        - Image clarity.
        - Distinctiveness of the food.
        - Visibility of ingredients.
        - Identification certainty.

        Condition confidence should reflect:
        - Visibility of abnormal features.
        - Clarity of charring, mold or discoloration.
        - Reliability of the visual assessment.

        Both confidence scores must be between 0 and 1.

        STEP 6: FINAL OUTPUT

        Return ONLY one valid JSON object.

        Use exactly this schema:

        {
            "label": "chicken biryani",
            "portion_unit": "1 plate",
            "estimated_weight_g": 250,
            "weight_confidence": 0.70,
            "confidence": 0.95,

            "visual_condition": "normal_appearance",
            "condition_severity": "none",
            "condition_confidence": 0.90,
            "condition_evidence": "No obvious visible signs of burning or spoilage.",

            "calories": 450,
            "protein_g": 25,
            "fat_g": 15,
            "carbs_g": 55,
            "fiber_g": 3
        }

        Requirements:
        - Return all specified keys.
        - All nutritional values must be numeric and non-negative.
        - Calories must be an integer.
        - estimated_weight_g must be a non-negative whole number in grams.
        - weight_confidence must be between 0 and 1.
        - Confidence values must be between 0 and 1.
        - Use only the permitted condition labels and severity values.
        - Return uncertain when visual evidence is insufficient.
        - Do not include markdown, explanations or additional keys.

        A food image cannot establish its actual expiry date
        or guarantee microbiological safety. Never claim otherwise.
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

        def safe_confidence(value):

            number = safe_float(value)

            if number > 1:
                raise ValueError("Confidence must be between 0 and 1.")

            return number

        valid_conditions = {
            "normal_appearance",
            "overcooked_or_burnt",
            "possible_spoilage_signs",
            "uncertain"
        }

        valid_severities = {
            "none",
            "mild",
            "moderate",
            "severe",
            "unknown"
        }

        condition = str(
            data.get("visual_condition", "uncertain")
        ).strip().lower()

        severity = str(
            data.get("condition_severity", "unknown")
        ).strip().lower()

        if condition not in valid_conditions:
            condition = "uncertain"

        if severity not in valid_severities:
            severity = "unknown"

        condition_confidence = safe_confidence(
            data.get("condition_confidence", 0)
        )

        return {
            "label": label,
            "portion_unit": str(
                data.get("portion_unit", "Unknown")
            ),
            "estimated_weight_g": int(
                safe_float(data.get("estimated_weight_g", 0))
            ),
            "weight_confidence": safe_confidence(
                data.get("weight_confidence", 0)
            ),
            "confidence": safe_confidence(data.get("confidence", 0)),
            "calories": safe_float(data.get("calories", 0)),
            "protein": safe_float(data.get("protein_g", 0)),
            "fat": safe_float(data.get("fat_g", 0)),
            "carbs": safe_float(data.get("carbs_g", 0)),
            "fiber": safe_float(data.get("fiber_g", 0)),
            "visual_condition": condition,
            "condition_severity": severity,
            "condition_confidence": condition_confidence,
            "condition_evidence": str(
                data.get("condition_evidence", "Insufficient visual evidence.")
            ),
            "expiration_status": "cannot_determine_from_image",
        }

    # Default response for empty images
    def _default_response(self):

        return {
            "label": "unknown",
            "portion_unit": "Unknown",
            "estimated_weight_g": 0,
            "weight_confidence": 0.0,
            "confidence": 0.0,
            "calories": 0.0,
            "protein": 0.0,
            "fat": 0.0,
            "carbs": 0.0,
            "fiber": 0.0,
            "visual_condition": "uncertain",
            "condition_severity": "unknown",
            "condition_confidence": 0.0,
            "condition_evidence": "Image unavailable.",
            "expiration_status": "cannot_determine_from_image"
        }
