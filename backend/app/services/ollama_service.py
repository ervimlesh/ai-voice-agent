from typing import Iterable

import httpx

from app.core.config import Settings
from app.core.language import contains_non_english_script
from app.schemas.chat import ChatMessage


class OllamaService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.chat_url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"

    async def _chat(
        self,
        messages: list[dict[str, str]],
        options: dict | None = None,
    ) -> str:
        payload = {
            "model": self.settings.ollama_model,
            "messages": messages,
            "stream": False,
        }
        # Ollama runtime options (e.g. num_predict to cap output length,
        # temperature). Passed through verbatim when provided.
        if options:
            payload["options"] = options
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(self.chat_url, json=payload)
            response.raise_for_status()
            data = response.json()
        return data.get("message", {}).get("content", "").strip()

    def _safe_history(self, history: Iterable[ChatMessage]) -> list[dict[str, str]]:
        filtered = [
            {"role": item.role, "content": item.content}
            for item in history
            if item.role in {"user", "assistant"} and item.content.strip()
        ]
        return filtered[-self.settings.max_history_messages :]

    async def ask(
        self,
        user_text: str,
        history: list[ChatMessage] | None = None,
        speaker_role: str = "Patient",
        rag_suggestions: list[str] | None = None,
    ) -> str:
        """
        Enhanced ask() with medical context awareness and doctor suggestions.

        Args:
            user_text: The user's speech transcript
            history: Conversation history
            speaker_role: "Doctor" or "Patient" — determines system prompt
            rag_suggestions: Suggested follow-up questions from medical knowledge base

        Returns:
            AI assistant response
        """
        history = history or []
        rag_suggestions = rag_suggestions or []

        # Build context-aware system prompt
        if speaker_role == "Patient":
            system_prompt = self._build_patient_system_prompt(rag_suggestions)
        else:
            system_prompt = self._build_doctor_system_prompt()

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self._safe_history(history))
        messages.append({"role": "user", "content": user_text})

        reply = await self._chat(messages)
        return reply.strip()

    def _build_patient_system_prompt(self, rag_suggestions: list[str]) -> str:
        """Build system prompt for patient utterances with RAG doctor suggestions."""
        suggestions_text = ""
        if rag_suggestions:
            numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(rag_suggestions))
            suggestions_text = f"\n\nSUGGESTED FOLLOW-UP QUESTIONS (for doctor to ask):\n{numbered}"

        return f"""You are a helpful AI assistant in a real-time doctor-patient consultation.
The current speaker is a PATIENT.

If the patient is discussing symptoms or health concerns:
- Briefly summarize what they said
- Provide 2-3 relevant follow-up questions for the doctor to consider
{suggestions_text}

If the patient is discussing non-medical topics (like general conversation):
- Respond naturally and helpfully
- Keep it conversational and brief

RULES:
- Always reply in English
- Be concise (2-4 sentences maximum)
- Do not diagnose; only assess symptoms if health-related
- Never assume information not explicitly stated
- Match the tone to the topic (clinical for health, conversational for general topics)"""

    def _build_doctor_system_prompt(self) -> str:
        """Build system prompt for doctor utterances."""
        return """You are a helpful AI assistant supporting a real-time doctor-patient consultation.
The current speaker is a DOCTOR.

If the doctor is discussing clinical matters:
- Briefly acknowledge what they said
- Suggest what to address or clarify next

If the doctor is discussing non-medical topics:
- Respond naturally and helpfully

RULES:
- Always reply in English
- Keep response to 1-3 sentences maximum
- Be concise and practical
- Match the tone to the topic"""

    async def ask_multilingual(self, user_text: str, detected_language: str, history: list[ChatMessage] | None = None) -> str:
        history = history or []
        language_name = self._get_language_name(detected_language)
        
        messages = [
            {
                "role": "system",
                "content": f"""
You are a helpful AI assistant that responds in the user's language.

The user spoke in: {language_name} (language code: {detected_language})

IMPORTANT RULES:
1. Understand the user's message in {language_name}.
2. Reply ONLY in {language_name}.
3. Keep your answer clear, concise, and natural.
4. Do not use English unless the user's language is English.
5. Match the user's language exactly.

Your response must be in {language_name} only.
""",
            }
        ]
        messages.extend(self._safe_history(history))
        messages.append(
            {
                "role": "user",
                "content": f"User asked in {language_name}:\n\n{user_text}\n\nReply in {language_name} only.",
            }
        )
        reply = await self._chat(messages)
        return reply.strip()

    def _get_language_name(self, language_code: str) -> str:
        language_map = {
            "en": "English",
            "hi": "Hindi",
            "te": "Telugu",
            "ta": "Tamil",
            "kn": "Kannada",
            "ml": "Malayalam",
            "mr": "Marathi",
            "bn": "Bengali",
            "ur": "Urdu",
            "gu": "Gujarati",
            "pa": "Punjabi",
            "or": "Odia",
            "as": "Assamese",
        }
        return language_map.get(language_code, language_code.upper())

    async def ask_generic_english(self, user_text: str, history: list[ChatMessage] | None = None, detected_language: str = "en") -> str:
        """
        ROBUST method: accepts input in any language but responds ONLY in English.
        Uses a two-step approach:
        1. If input is non-English, translate to English first
        2. Process and respond in English only

        Args:
            user_text: User's message (can be in any language)
            history: Conversation history
            detected_language: Language detected by Whisper

        Returns:
            Response in English only
        """
        history = history or []

        # Step 1: If input is in non-English language, translate to English first
        english_user_text = user_text
        if detected_language != "en" and contains_non_english_script(user_text):
            print(f"🔄 Input is in {detected_language}, translating to English first...")
            english_user_text = await self._translate_to_english(user_text)
            print(f"✅ Translated to English: {english_user_text}")

        # Step 2: Process with a CLEAR, DIRECTIVE system prompt
        system_prompt = """You are a helpful, informative AI assistant.

Your task:
- Understand what the user is asking
- Provide helpful, accurate information
- ALWAYS respond ONLY in English
- Be clear, concise, and natural
- Never apologize for using English
- Just provide the answer they need

Remember: Your response MUST be 100% in English. No exceptions."""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self._safe_history(history))
        messages.append({"role": "user", "content": english_user_text})

        reply = await self._chat(messages)

        # Step 3: Guarantee English output - force translate if needed
        if contains_non_english_script(reply):
            print(f"⚠️ Response contains non-English, force translating...")
            reply = await self.force_english_reply(reply)

        return reply.strip()

    async def _translate_to_english(self, text: str) -> str:
        """Translate any non-English text to English."""
        messages = [
            {
                "role": "system",
                "content": "You are a professional translator. Translate the input to English. Output ONLY the English translation, nothing else."
            },
            {
                "role": "user",
                "content": text
            }
        ]
        return await self._chat(messages)

    async def force_english_reply(self, reply_text: str) -> str:
        if not reply_text:
            return ""

        # Check if response contains non-English script
        has_non_english = contains_non_english_script(reply_text)

        if not has_non_english:
            # Already in English, return as-is
            return reply_text.strip()

        # Response is in another language - TRANSLATE to English
        print(f"🔄 Translating non-English response to English...")
        messages = [
            {
                "role": "system",
                "content": """You are a professional translator.
Translate the given text to English.
Provide ONLY the English translation.
No explanations, no extra text.
Just the English version of the input text.""",
            },
            {
                "role": "user",
                "content": f"Translate this to English:\n\n{reply_text}"
            },
        ]
        translated = await self._chat(messages)
        print(f"✅ Translated to English: {translated[:100]}...")
        return translated.strip()
