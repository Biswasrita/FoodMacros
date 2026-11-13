import json
import os
from google import genai
from PIL import Image


class LLMFoodClassifier:

  def __init__(self, api_key=None):
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
      raise ValueError("API Key is missing.")

    self.client = genai.Client(api_key=key)

  def predict(self, crop_np):
    if crop_np is None or crop_np.size == 0:
      return self._default_response()

    pil_image = Image.fromarray(crop_np)

    # Prompt engineered to request visual portion and macronutrient estimation
    prompt = """
        Analyze this cropped image of food.
        1. Identify the specific food item.
        2. Visually estimate the portion unit/container (e.g., '1 bowl', '1 plate', '1 slice', '1 cup', '1 piece', '1 scoop', '1 roll').
        3. Estimate the macronutrients based on this specific visual portion.

        Return ONLY a valid JSON object in this exact format with no markdown, backticks, or extra text:
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
    try:
      response = self.client.models.generate_content(
          model="gemini-2.5-flash",
          contents=[prompt, pil_image],
      )
    except Exception as api_error:
      print(f"Primary model failed ({api_error}). Trying fallback...")
      try:
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, pil_image],
        )
      except Exception as fallback_error:
        print(f"Gemini API Error: {fallback_error}")
        return self._default_response()

    try:
      response_text = response.text
      if not response_text:
        return self._default_response()

      clean_text = (
          response_text.replace("```json", "").replace("```", "").strip()
      )
      data = json.loads(clean_text)

      return {
          "label": str(data.get("label", "unknown")).strip().lower(),
          "confidence": float(data.get("confidence", 0.0)),
          "calories": float(data.get("calories", 0.0)),
          "protein": float(data.get("protein_g", 0.0)),
          "fat": float(data.get("fat_g", 0.0)),
          "carbs": float(data.get("carbs_g", 0.0)),
          "fiber": float(data.get("fiber_g", 0.0)),
      }

    except (json.JSONDecodeError, ValueError, TypeError) as parse_error:
      print(
          f"Failed to parse LLM response: {response.text} | Error:"
          f" {parse_error}"
      )
      return self._default_response()

  def _default_response(self):
    return {
        "label": "unknown",
        "confidence": 0.0,
        "calories": 0.0,
        "protein": 0.0,
        "fat": 0.0,
        "carbs": 0.0,
        "fiber": 0.0,
    }