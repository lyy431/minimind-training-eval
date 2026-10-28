"""Existing demo tool definitions and executors, preserved verbatim."""
import random
from trainer.trainer_utils import safe_math_eval

# 工具定义
# 与 agent_rl_math.jsonl 训练数据里的定义逐字一致（不带 "type" 字段）：模型对工具描述的措辞很敏感，
# 用原来的简写描述时 bb_rl 的数学全对率从 81.8% 掉到 54.5%（见 my_eval/README.md）。
# random_number / text_length 训练数据里没有，沿用原描述。
TOOLS = [
    {"function": {"name": "calculate_math", "description": "计算数学表达式的结果，支持加减乘除、幂运算", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式，如123+456、2**10"}}, "required": ["expression"]}}},
    {"function": {"name": "get_current_time", "description": "获取当前日期和时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "时区，如Asia/Shanghai", "default": "Asia/Shanghai"}}, "required": []}}},
    {"function": {"name": "random_number", "description": "生成随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}}, "required": ["min", "max"]}}},
    {"function": {"name": "text_length", "description": "计算文本长度", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"function": {"name": "unit_converter", "description": "进行单位换算，支持长度、重量、温度", "parameters": {"type": "object", "properties": {"value": {"type": "number", "description": "要转换的数值"}, "from_unit": {"type": "string", "description": "源单位"}, "to_unit": {"type": "string", "description": "目标单位"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"function": {"name": "get_current_weather", "description": "获取指定城市的当前天气信息", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "城市名称，如北京、上海"}}, "required": ["location"]}}},
    {"function": {"name": "get_exchange_rate", "description": "查询两种货币之间的汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string", "description": "源货币代码，如USD、CNY、EUR"}, "to_currency": {"type": "string", "description": "目标货币代码"}}, "required": ["from_currency", "to_currency"]}}},
    {"function": {"name": "translate_text", "description": "将文本翻译成目标语言", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要翻译的文本"}, "target_language": {"type": "string", "description": "目标语言，如english、chinese、japanese"}}, "required": ["text", "target_language"]}}},
]

TOOL_SHORT_NAMES = {
    'calculate_math': '数学', 'get_current_time': '时间', 'random_number': '随机',
    'text_length': '字数', 'unit_converter': '单位', 'get_current_weather': '天气',
    'get_exchange_rate': '汇率', 'translate_text': '翻译'
}

def execute_tool(tool_name, args):
    import datetime
    try:
        if tool_name == 'calculate_math':
            return {"result": safe_math_eval(args.get('expression', '0'))}
        elif tool_name == 'get_current_time':
            tz = args.get('timezone', 'Asia/Shanghai')
            return {"result": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        elif tool_name == 'random_number':
            return {"result": random.randint(args.get('min', 0), args.get('max', 100))}
        elif tool_name == 'text_length':
            return {"result": len(args.get('text', ''))}
        elif tool_name == 'unit_converter':
            return {"result": f"{args.get('value', 0)} {args.get('from_unit', '')} = ? {args.get('to_unit', '')}"}
        elif tool_name == 'get_current_weather':
            return {"result": f"{args.get('location', args.get('city', 'Unknown'))}: 晴, 7~10°C"}
        elif tool_name == 'get_exchange_rate':
            return {"result": f"1 {args.get('from_currency', 'USD')} = 7.2 {args.get('to_currency', 'CNY')}"}
        elif tool_name == 'translate_text':
            return {"result": f"[翻译结果]: hello world"}
        return {"result": "Unknown tool"}
    except Exception as e:
        return {"error": str(e)}

