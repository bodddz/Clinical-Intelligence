import os
import sys

# Ensure backend directory is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# 1. Test ValueError when GEMINI_API_KEY is not defined
os.environ.pop("GEMINI_API_KEY", None)
from pipeline import get_gemini_api_key

try:
    get_gemini_api_key()
    print("FAILED: Did not raise ValueError when GEMINI_API_KEY missing")
    sys.exit(1)
except ValueError as e:
    print(f"PASSED: Correctly raised ValueError when GEMINI_API_KEY missing -> {e}")

# 2. Test get_gemini_api_key when defined
os.environ["GEMINI_API_KEY"] = "test_valid_key_12345"
key = get_gemini_api_key()
assert key == "test_valid_key_12345", f"Expected test_valid_key_12345, got {key}"
print("PASSED: Correctly retrieved GEMINI_API_KEY from environment")

print("ALL SECURITY UNIT TESTS COMPLETED SUCCESSFULLY!")
