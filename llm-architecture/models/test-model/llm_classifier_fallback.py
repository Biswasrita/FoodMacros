
import os
import re
import json
import io
import time
import base64
import math
import requests

from google import genai
from PIL import Image


class LLMFoodClassifier:

    def __init__(self, api_key=None):

        gemini_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY")

        if not gemini_key:
            raise ValueError("Gemini API Key is missing.")

        self.client = genai.Client(api_key=gemini_key)

        # Primary model
        self.gemini_model = "gemini-3.8-flash"

        # OpenRouter fallback models
        self.openrouter_models = [
            "nvidia/nemotron-3.5-super-vl:free",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            "google/gemma-4-26b-a4b-it:free"
        ]

        self.openrouter_url = (
            "https://openrouter.ai/api/v1/chat/completions"
        )

        self.session = requests.Session()

    # --------------------------------
    # MAIN PREDICTION FUNCTION
    # --------------------------------

    def predict(self, crop_np):

        if crop_np is None or crop_np.size == 0:
            return self._default_response()

        pil_image = Image.fromarray(crop_np).convert("RGB")

        prompt = """
        Analyze this cropped image of food.

        Identify:
        1. The specific food item.
        2. The approximate portion size.
        3. Estimated nutritional values for the visible portion.

        Estimate calories and macronutrients based on the visible
        quantity. Do not assume a standard serving if the portion
        appears different.

        Return ONLY a valid JSON object:

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

        All nutritional values must be numeric.
        Do not include markdown or explanations.
        """

        # 1. PRIMARY: GEMINI
        for attempt in range(2):

            try:

                print(
                    f"Trying Gemini API "
                    f"(attempt {attempt + 1}/2)..."
                )

                response = self.client.models.generate_content(
                    model=self.gemini_model,
                    contents=[prompt, pil_image],
                )

                if not response.text:
                    raise ValueError(
                        "Gemini returned an empty response."
                    )

                result = self._parse_response(response.text)

                if result is None:
                    raise ValueError(
                        "Gemini returned invalid JSON."
                    )

                print("Gemini succeeded.")

                return result

            except Exception as error:

                print(
                    f"Gemini attempt {attempt + 1} failed: {error}"
                )

                if attempt == 0:
                    time.sleep(2)

        # 2. FALLBACK: OPENROUTER
        if not self.openrouter_key:

            print("OpenRouter API key is missing.")

            return self._default_response()

        # Prepare image once for all fallback models
        try:

            pil_image.thumbnail((1024, 1024))

            image_buffer = io.BytesIO()

            pil_image.save(
                image_buffer,
                format="JPEG",
                quality=85
            )

            image_base64 = base64.b64encode(
                image_buffer.getvalue()
            ).decode("utf-8")

            image_url = (
                f"data:image/jpeg;base64,{image_base64}"
            )

        except Exception as error:

            print(f"Image preparation failed: {error}")

            return self._default_response()

        # TRY OPENROUTER MODELS SEQUENTIALLY
        for model in self.openrouter_models:

            try:

                print(f"Trying OpenRouter model: {model}")

                headers = {
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8501",
                    "X-Title": "FoodMacros"
                }

                payload = {
                    "model": model,

                    "messages": [
                        {
                            "role": "user",

                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",

                                    "image_url": {
                                        "url": image_url
                                    }
                                }
                            ]
                        }
                    ],

                    "temperature": 0.2,
                    "max_tokens": 500
                }

                response = self.session.post(
                    self.openrouter_url,
                    headers=headers,
                    json=payload,
                    timeout=(15, 90)
                )

                # Check API errors
                if response.status_code != 200:

                    print(
                        f"OpenRouter error ({response.status_code}): "
                        f"{response.text[:1500]}"
                    )

                    # Try the next model
                    continue

                result = response.json()

                choices = result.get("choices", [])

                if not choices:
                    raise ValueError(
                        f"No choices returned: {result}"
                    )

                response_text = (
                    choices[0]
                    .get("message", {})
                    .get("content", "")
                )

                # Handle text returned as a list
                if isinstance(response_text, list):

                    response_text = "".join(
                        part.get("text", "")
                        for part in response_text
                        if isinstance(part, dict)
                    )

                if not response_text:
                    raise ValueError(
                        "Model returned an empty response."
                    )

                parsed_result = self._parse_response(
                    response_text
                )

                if parsed_result is None:

                    raise ValueError(
                        "Model returned invalid JSON."
                    )

                print(f"OpenRouter succeeded: {model}")

                return parsed_result

            except Exception as error:

                print(
                    f"OpenRouter model {model} failed: {error}"
                )

                continue

        # 3. ALL MODELS FAILED
        print("All AI models failed.")

        return self._default_response()

    # COMMON JSON PARSER
    def _parse_response(self, response_text):

        try:

            clean_text = response_text.strip()

            # Remove markdown code fences
            clean_text = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                clean_text,
                flags=re.IGNORECASE
            ).strip()

            # Extract JSON object
            start = clean_text.find("{")
            end = clean_text.rfind("}")

            if start == -1 or end == -1:
                raise ValueError("No JSON object found.")

            data = json.loads(
                clean_text[start:end + 1]
            )

            label = str(
                data.get("label", "unknown")
            ).strip().lower()

            if not label or label == "unknown":
                raise ValueError("Food label is missing.")

            def safe_float(value):

                number = float(value)

                if not math.isfinite(number) or number < 0:
                    raise ValueError(
                        "Invalid nutritional value."
                    )

                return number

            return {
                "label": label,

                "confidence": safe_float(
                    data.get("confidence", 0.0)
                ),

                "calories": safe_float(
                    data.get("calories", 0.0)
                ),

                "protein": safe_float(
                    data.get("protein_g", 0.0)
                ),

                "fat": safe_float(
                    data.get("fat_g", 0.0)
                ),

                "carbs": safe_float(
                    data.get("carbs_g", 0.0)
                ),

                "fiber": safe_float(
                    data.get("fiber_g", 0.0)
                )
            }

        except (json.JSONDecodeError, ValueError, TypeError) as error:

            print(f"JSON parsing failed: {error}")

            print(f"Raw response: {response_text}")

            return None

    # DEFAULT RESPONSE
    def _default_response(self):

        return {
            "label": "unknown",
            "confidence": 0.0,
            "calories": 0.0,
            "protein": 0.0,
            "fat": 0.0,
            "carbs": 0.0,
            "fiber": 0.0
        }
