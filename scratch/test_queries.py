import urllib.request
import json

def test_query(text):
    req = urllib.request.Request(
        'http://127.0.0.1:8000/api/query',
        data=json.dumps({'query': text}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    res = urllib.request.urlopen(req).read().decode('utf-8')
    return json.loads(res)

cases = [
    ("Smalltalk", "how are you doing today?"),
    ("Gibberish", "asdfghjk1234"),
    ("Ambiguity Gate 0", "what is the recurrence rate?"),
    ("Arabic Greeting", "السلام عليكم ازيك"),
    ("Clinical PWNE", "What was the ASM treatment protocol for PWNE patients?"),
]

for label, query in cases:
    res = test_query(query)
    print(f"=== {label} : '{query}' ===")
    print("Answer:", res.get("answer"))
    print("Confidence:", res.get("confidence_level"))
    print("Nuance:", res.get("clinical_nuance"))
    print("Quotes:", res.get("grounded_quotes"))
    print("Telemetry:", res.get("telemetry"))
    print()
