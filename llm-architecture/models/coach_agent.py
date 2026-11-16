import os
from google import genai

class FudAICoach:
    def __init__(self, api_key=None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=key)

    def chat(self, user_message, context_data, history):
        system_instruction = f"""
        You are Fud AI Coach, an empathetic, science-based health and nutrition partner.
        You have full access to the user's live context:
        - Today's Nutrition Summary: {context_data.get('daily_summary')}
        - Active Fasting Status: {context_data.get('fasting')}
        - Logged Workouts: {context_data.get('workouts')}
        
        Provide actionable, concise, and scannable advice. Never assume missing meals mean fasting.
        """
        
        contents = [system_instruction]
        for msg in history[-6:]: # Keep recent conversation turn context
            contents.append(f"{msg['role'].capitalize()}: {msg['content']}")
        contents.append(f"User: {user_message}")

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents
            )
            return response.text
        except Exception as e:
            return f"Coach unavailable: {e}"