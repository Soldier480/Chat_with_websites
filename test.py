import os
import time
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 1. List models available to your key
r = requests.get(
    "https://generativelanguage.googleapis.com/v1beta/models",
    params={"key": api_key, "pageSize": 200},
)
r.raise_for_status()
names = [
    m["name"].replace("models/", "")
    for m in r.json().get("models", [])
    if "generateContent" in m.get("supportedGenerationMethods", [])
]
print("Available models:")
for n in names:
    print(" ", n)

# 2. Test Flash models, retrying on 503/429
flash = [n for n in names if "flash" in n and "image" not in n and "tts" not in n][:6]
for m in flash:
    for attempt in range(3):
        try:
            llm = ChatGoogleGenerativeAI(model=m, google_api_key=api_key)
            out = llm.invoke("Say hello in one short sentence.")
            print("WORKS:", m, "->", out.content)
            break
        except Exception as e:
            msg = str(e)
            if "503" in msg or "429" in msg:
                time.sleep(5)
                continue
            print("FAIL: ", m, "->", msg[:100].replace("\n", " "))
            break
    else:
        print("BUSY: ", m)