"""
Unit tests to verify that the system always responds in English,
regardless of input language.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import Settings
from app.services.ollama_service import OllamaService
from app.core.language import contains_non_english_script
from app.schemas.chat import ChatMessage


async def test_english_only_hindi_input():
    """Test: Hindi input → English response"""
    print("\n" + "="*70)
    print("TEST 1: Hindi Input → English Response")
    print("="*70)

    settings = Settings()
    ollama_service = OllamaService(settings)

    hindi_text = "मेरे सिर में दर्द है और मुझे बुखार है"  # "I have headache and fever"
    print(f"Input (Hindi): {hindi_text}")
    print(f"Has non-English script: {contains_non_english_script(hindi_text)}")

    try:
        response = await ollama_service.ask_generic_english(hindi_text, [])
        print(f"\nResponse: {response}")
        print(f"Response has non-English script: {contains_non_english_script(response)}")

        if not contains_non_english_script(response):
            print("✅ PASS: Response is in English")
            return True
        else:
            print("❌ FAIL: Response contains non-English script")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


async def test_english_only_tamil_input():
    """Test: Tamil input → English response"""
    print("\n" + "="*70)
    print("TEST 2: Tamil Input → English Response")
    print("="*70)

    settings = Settings()
    ollama_service = OllamaService(settings)

    tamil_text = "என் தலை வலிக்கிறது"  # "My head hurts"
    print(f"Input (Tamil): {tamil_text}")
    print(f"Has non-English script: {contains_non_english_script(tamil_text)}")

    try:
        response = await ollama_service.ask_generic_english(tamil_text, [])
        print(f"\nResponse: {response}")
        print(f"Response has non-English script: {contains_non_english_script(response)}")

        if not contains_non_english_script(response):
            print("✅ PASS: Response is in English")
            return True
        else:
            print("❌ FAIL: Response contains non-English script")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


async def test_english_only_telugu_input():
    """Test: Telugu input → English response"""
    print("\n" + "="*70)
    print("TEST 3: Telugu Input → English Response")
    print("="*70)

    settings = Settings()
    ollama_service = OllamaService(settings)

    telugu_text = "నాకు జ్వరం ఉంది"  # "I have fever"
    print(f"Input (Telugu): {telugu_text}")
    print(f"Has non-English script: {contains_non_english_script(telugu_text)}")

    try:
        response = await ollama_service.ask_generic_english(telugu_text, [])
        print(f"\nResponse: {response}")
        print(f"Response has non-English script: {contains_non_english_script(response)}")

        if not contains_non_english_script(response):
            print("✅ PASS: Response is in English")
            return True
        else:
            print("❌ FAIL: Response contains non-English script")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


async def test_force_english_reply():
    """Test: force_english_reply converts non-English to English"""
    print("\n" + "="*70)
    print("TEST 4: Force English Reply - Translate Hindi to English")
    print("="*70)

    settings = Settings()
    ollama_service = OllamaService(settings)

    hindi_response = "मुझे लगता है कि आपको सर्दी हो सकती है। पानी पीएं और आराम करें।"
    print(f"Hindi Response: {hindi_response}")
    print(f"Has non-English script: {contains_non_english_script(hindi_response)}")

    try:
        english_response = await ollama_service.force_english_reply(hindi_response)
        print(f"\nTranslated Response: {english_response}")
        print(f"Has non-English script: {contains_non_english_script(english_response)}")

        if not contains_non_english_script(english_response):
            print("✅ PASS: Translated to English")
            return True
        else:
            print("❌ FAIL: Translation still contains non-English")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


async def test_english_input_english_output():
    """Test: English input → English response"""
    print("\n" + "="*70)
    print("TEST 5: English Input → English Response")
    print("="*70)

    settings = Settings()
    ollama_service = OllamaService(settings)

    english_text = "I have a headache and fever"
    print(f"Input (English): {english_text}")
    print(f"Has non-English script: {contains_non_english_script(english_text)}")

    try:
        response = await ollama_service.ask_generic_english(english_text, [])
        print(f"\nResponse: {response}")
        print(f"Response has non-English script: {contains_non_english_script(response)}")

        if not contains_non_english_script(response):
            print("✅ PASS: Response is in English")
            return True
        else:
            print("❌ FAIL: Response contains non-English script")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


async def run_all_tests():
    """Run all unit tests"""
    print("\n" + "="*70)
    print("ENGLISH-ONLY RESPONSE UNIT TESTS")
    print("="*70)
    print("Testing that system accepts input in any language")
    print("but ALWAYS responds in English only")
    print("="*70)

    results = []

    # Test 1: Hindi
    results.append(await test_english_only_hindi_input())

    # Test 2: Tamil
    results.append(await test_english_only_tamil_input())

    # Test 3: Telugu
    results.append(await test_english_only_telugu_input())

    # Test 4: Force English Reply
    results.append(await test_force_english_reply())

    # Test 5: English
    results.append(await test_english_input_english_output())

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✅ ALL TESTS PASSED - English-only output is working!")
    else:
        print(f"❌ {total - passed} tests failed")

    print("="*70 + "\n")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
