import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# =========================================================
# OpenRouter
# =========================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODEL = "openrouter/free"

# =========================================================
# Tool本体
# =========================================================

def get_weather(city: str) -> str:

    weather = {
        "東京": "晴れ 30℃",
        "大阪": "曇り 29℃",
        "札幌": "雨 22℃",
    }

    return weather.get(
        city,
        f"{city}のデータはありません"
    )

# =========================================================
# LLMに見せるToolの定義
# =========================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "指定された都市の現在の天気を取得する",
            "parameters": {
                "type": "object",
                "properties":{
                    "city": {
                        "type": "string",
                        "desctiption": "都市名。例: 東京"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# =========================================================
# Tool名 → Python関数
# =========================================================

TOOL_HANDLERS = {
    "get_weather": get_weather
}

# =========================================================
# Agent Loop
# =========================================================

def run_agent(user_input: str):

    messages = [
        {
            "role": "system",
            "content": """
あなたはシンプルなAI Agentです。

天気について質問された場合は、
必ず get_weather tool を使用してください。

toolの結果を受け取ったら、
その結果を使ってユーザーに回答してください。
"""
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    

    step = 1

    while True:

        print(f"\nLoop {step}")
        print(messages[-1])

        print(f"\n===== LOOP {step} ====")
        print("[LLMを呼び出す]")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        message = response.choices[0].message

        print("\Message")
        print(response)

        # =================================================
        # Tool Call がない
        # → 最終回答
        # =================================================

        if not message.tool_calls:

            print("\n[Tool Callなし]")
            print("-> Agent Loop終了")

            print("\nAgent:")
            print(message.content)

            break

        # =================================================
        # Tool Call がある
        # =================================================

        print("\n[Tool Callあり]")

        # LLMが返したassistantメッセージを履歴に追加
        messages.append(message)

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            print("\nTool:")
            print(tool_name)

            print("\nArguments:")
            print(arguments)

            # Tool 実行

            tool_function = TOOL_HANDLERS[tool_name]

            result = tool_function(**arguments)

            print("\nTool Result:")
            print(result)

            # Tool ResultをLLMに返す

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                }
            )

            step += 1


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":

    user_input = input("You: ")

    run_agent(user_input)       
        