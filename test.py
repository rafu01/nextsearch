from google import genai

client = genai.Client(http_options={"api_version": "v1"})

for m in client.models.list():
    if "embed" in m.name.lower():
        print(m.name)
