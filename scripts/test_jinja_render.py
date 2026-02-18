from jinja2 import Environment, FileSystemLoader

# 测试数据
messages = [
    {"role": "system", "content": "You are a helpful weather assistant."},
    {"role": "user", "content": "What's the weather in London?"},
    # 函数调用
    {"role": "assistant", "content": '<tool_call>{"name": "get_weather", "arguments": "{"city": "London"}"}</tool_call>'},
    # 函数调用结果
    {"role": "tool_response", "content": [{"name": "get_weather", "response": "The weather is clear."},]},
]

tools = [
    {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "arguments": {"city": {"type": "string"}}
    },
],

tool_calls = [
    {"name": "get_weather", "response": "tool_call response ok"}
]

# 置模板目录
env = Environment(loader=FileSystemLoader("./scripts"))
template = env.get_template("chat_template.jinja")

# 渲染模板
output = template.render(messages=messages, tools=tools)
# print(repr(output))
print(output)