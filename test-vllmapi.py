from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://10.96.176.208:8000/v1")

resp = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[{"role": "user", "content": "你好，请介绍一下你自己"}],
    temperature=0.7
)

print(resp.choices[0].message.content)
