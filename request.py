import requests
import json

url = "http://localhost:11434/api/chat"

payload = {
    # 如果你刚才删了 qwen3-vl-local，记得这里换成你现在正在用的模型名
    "model": "qwen3-vl-local", 
    "messages": [
        {
            "role": "user",
            "content": "你好，请用一段话介绍一下人工智能。"
        }
    ],
    # 关键修改 1：告诉 Ollama 我们需要流式返回
    "stream": True 
}

print("正在发送请求，等待模型开口...\n")

try:
    # 关键修改 2：告诉 requests 库保持连接，采用流式接收
    response = requests.post(url, json=payload, stream=True)
    
    if response.status_code == 200:
        # 关键修改 3：一行一行地读取返回的数据，而不是等全部结束
        for line in response.iter_lines():
            if line:
                # 解析每一行的 JSON 数据
                data = json.loads(line.decode('utf-8'))
                
                # 提取文字并立刻打印到屏幕上 (flush=True 强制立即刷新输出)
                if 'message' in data and 'content' in data['message']:
                    print(data['message']['content'], end='', flush=True)
        print("\n\n[回答结束]")
    else:
        print(f"请求失败，状态码: {response.status_code}")

except Exception as e:
    print(f"发生错误: {e}")