from openai import OpenAI

client = OpenAI(
    api_key="sk-v46hWyxfMF72b2T4603d6c0638Cf45078c027d1cE3E8C33a",
    base_url="https://api.apiyi.com/v1"  # 只到 /v1
)

resp = client.chat.completions.create(
    model="deepseek-chat",  # 这里改成 deepseek 模型名
    messages=[
        {"role": "user", "content": "Hello! What's your name?"}
    ]
)
print(resp.choices[0].message.content)