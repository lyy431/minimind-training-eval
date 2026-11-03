import random
import re
import json
import os
import sys
import copy
from threading import Thread

import torch
import numpy as np
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from best_of_n import best_of_n, INFERENCE_LOCK
from demo_best_of_support import SafeRewardSelector, trim_context

st.set_page_config(page_title="MiniMind", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        /* 添加操作按钮样式 */
        .stButton button {
            border-radius: 50% !important;  /* 改为圆形 */
            width: 32px !important;         /* 固定宽度 */
            height: 32px !important;        /* 固定高度 */
            padding: 0 !important;          /* 移除内边距 */
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #666 !important;         /* 更柔和的颜色 */
            margin: 5px 10px 5px 0 !important;  /* 调整按钮间距 */
        }
        .stButton button:hover {
            border-color: #999 !important;
            color: #333 !important;
            background-color: #f5f5f5 !important;
        }
        .stMainBlockContainer > div:first-child {
            margin-top: -50px !important;
        }
        .stApp > div:last-child {
            margin-bottom: -35px !important;
        }
        
        /* 重置按钮基础样式 */
        .stButton > button {
            all: unset !important;  /* 重置所有默认样式 */
            box-sizing: border-box !important;
            border-radius: 50% !important;
            width: 18px !important;
            height: 18px !important;
            min-width: 18px !important;
            min-height: 18px !important;
            max-width: 18px !important;
            max-height: 18px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #888 !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            margin: 0 2px !important;  /* 调整这里的 margin 值 */
        }

    </style>
""", unsafe_allow_html=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 多语言文本
LANG_TEXTS = {
    'zh': {
        'settings': '模型设定调整',
        'history_rounds': '历史对话轮次',
        'max_length': '最大生成长度',
        'temperature': '温度',
        'thinking': '思考',
        'tools': '工具',
        'language': '语言',
        'send': '给 MiniMind 发送消息',
        'disclaimer': 'AI 生成内容可能存在错误，请仔细核实',
        'think_tip': '自适应思考，目前多轮对话或Tool Call共存时思考不稳定',
        'tool_select': '工具选择（最多6个）',
    },
    'en': {
        'settings': 'Model Settings',
        'history_rounds': 'History Rounds',
        'max_length': 'Max Length',
        'temperature': 'Temperature',
        'thinking': 'Thinking',
        'tools': 'Tools',
        'language': 'Language',
        'send': 'Send a message to MiniMind',
        'disclaimer': 'AI-generated content may be inaccurate, please verify',
        'think_tip': 'Adaptive thinking; may be unstable with multi-turn or Tool Call',
        'tool_select': 'Tool Selection (max 6)',
    }
}

def get_text(key):
    lang = st.session_state.get('lang', 'en')
    return LANG_TEXTS.get(lang, {}).get(key, LANG_TEXTS['zh'].get(key, key))

from demo_tools import TOOLS, TOOL_SHORT_NAMES, execute_tool
from demo_tool_guard import execute_selected_tool


def process_assistant_content(content, is_streaming=False):
    # 处理tool_call标签，格式化显示
    if '<tool_call>' in content:
        def format_tool_call(match):
            try:
                tc = json.loads(match.group(1))
                name = tc.get('name', 'unknown')
                args = tc.get('arguments', {})
                return f'<div style="background: rgba(80, 110, 150, 0.20); border: 1px solid rgba(140, 170, 210, 0.30); padding: 10px 12px; border-radius: 12px; margin: 6px 0;"><div style="font-size:12px;opacity:.75;display:block;margin:0 0 6px 0;line-height:1;">ToolCalling</div><div><b>{name}</b>: {json.dumps(args, ensure_ascii=False)}</div></div>'
            except:
                return match.group(0)
        content = re.sub(r'<tool_call>(.*?)</tool_call>', format_tool_call, content, flags=re.DOTALL)
    
    # 流式生成且开启思考时，一开始就放到折叠里
    if is_streaming and st.session_state.get('enable_thinking', False) and '</think>' not in content and '<think>' not in content:
        m = re.search(r'(\n\n(?:我是|您好|你好)[^\n]*)', content)
        if m and m.start(1) > 5:
            i = m.start(1)
            think_part = content[:i]
            answer_part = content[i:]
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_part.strip()}</div></details>{answer_part}'
        elif len(content) > 5:
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">思考中...</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto; display: flex; flex-direction: column-reverse;"><div style="margin-bottom: auto;">{content.strip().replace(chr(10), "<br>")}</div></div></details>'

    if '<think>' in content and '</think>' in content:
        def format_think(match):
            think_content = match.group(2)
            if think_content.replace('\n', '').strip():  # 不是全换行
                return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_content.strip()}</div></details>'
            return ''
        content = re.sub(r'(<think>)(.*?)(</think>)', format_think, content, flags=re.DOTALL)

    if '<think>' in content and '</think>' not in content:
        def format_think_in_progress(match):
            tc = match.group(1)
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">思考中...</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto; display: flex; flex-direction: column-reverse;"><div style="margin-bottom: auto;">{tc.strip().replace(chr(10), "<br>")}</div></div></details>'
        content = re.sub(r'<think>(.*?)$', format_think_in_progress, content, flags=re.DOTALL)

    if '<think>' not in content and '</think>' in content:
        def format_think_no_start(match):
            think_content = match.group(1)
            if think_content.replace('\n', '').strip():
                return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_content.strip()}</div></details>'
            return ''
        content = re.sub(r'(.*?)</think>', format_think_no_start, content, flags=re.DOTALL)

    return content


@st.cache_resource(max_entries=1)
def load_model_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = model.half().eval().to(device)
    return model, tokenizer


@st.cache_resource
def load_reward_selector():
    return SafeRewardSelector()


def clear_chat_messages():
    del st.session_state.messages
    del st.session_state.chat_messages


def init_chat_messages():
    if "messages" in st.session_state:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant":
                st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{message["content"]}</div></div>',
                    unsafe_allow_html=True)

    else:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    return st.session_state.messages

def regenerate_answer(index):
    st.session_state.messages.pop()
    st.session_state.chat_messages.pop()
    st.rerun()


# 动态扫描模型目录
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {}
for d in sorted(os.listdir(script_dir), reverse=True):
    full_path = os.path.join(script_dir, d)
    if os.path.isdir(full_path) and not d.startswith('.') and not d.startswith('_'):
        if any(f.endswith(('.bin', '.safetensors', '.pt')) or os.path.exists(os.path.join(full_path, 'model.safetensors.index.json')) for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))):
            MODEL_PATHS[d] = [d, d]
if not MODEL_PATHS:
    MODEL_PATHS = {"No models found": ["", "No models"]}

# 模型选择
selected_model = st.sidebar.selectbox('Model', list(MODEL_PATHS.keys()), index=0)
model_path = MODEL_PATHS[selected_model][0]
slogan = f"我是 {MODEL_PATHS[selected_model][1]}，有什么可以帮你的？" if st.session_state.get('lang', 'en') == 'zh' else f"I am {MODEL_PATHS[selected_model][1]}, how can I help you?"

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)

# 语言选择
lang_options = {'中文': 'zh', 'English': 'en'}
current_lang = st.session_state.get('lang', 'en')
lang_index = 0 if current_lang == 'zh' else 1
lang_label = st.sidebar.radio('Language / 语言', list(lang_options.keys()), index=lang_index, horizontal=True)
if lang_options[lang_label] != current_lang:
    st.session_state.lang = lang_options[lang_label]
    st.rerun()

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)

# 参数设置
st.session_state.history_chat_num = st.sidebar.slider(get_text('history_rounds'), 0, 8, 0, step=2)
st.session_state.max_new_tokens = st.sidebar.slider(get_text('max_length'), 256, 8192, 384 if selected_model in ('MiniMind-YuyangLU-v4', 'MiniMind-YuyangLU-v5') else 8192, step=1)
st.session_state.temperature = st.sidebar.slider(get_text('temperature'), 0.6, 1.2, 0.90, step=0.01)

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)

# 功能开关
st.session_state.enable_thinking = st.sidebar.checkbox(get_text('thinking'), value=False, help=get_text('think_tip'))
st.session_state.best_of_enabled = st.sidebar.checkbox(
    '多生成几个回答，再挑一个' if lang_options[lang_label] == 'zh' else 'Generate several answers and select one',
    value=False)
st.session_state.candidate_count = 1
if st.session_state.best_of_enabled:
    st.session_state.candidate_count = st.sidebar.select_slider(
        '候选数量 / Candidates', options=[2, 4], value=4)
    st.sidebar.caption('由本地奖励模型挑选，耗时更长，不保证答得更对。本模式关闭思考，每个候选最多生成512个词元/轮、3轮；过长或未完成的回答不参与挑选。'
        if lang_options[lang_label] == 'zh' else
        'A local reward model selects an answer. Slower; accuracy is not guaranteed. Thinking is off in this mode. Each candidate is limited to 512 tokens/turn and 3 turns. Incomplete or overlong answers are excluded.')
st.session_state.selected_tools = []
with st.sidebar.expander(get_text('tools')):
    st.caption(get_text('tool_select'))
    selected_count = sum(1 for tool in TOOLS if st.session_state.get(f"tool_{tool['function']['name']}", False))
    for tool in TOOLS:
        name = tool['function']['name']
        short_name = TOOL_SHORT_NAMES.get(name, name)
        checked = st.checkbox(short_name, key=f"tool_{name}", disabled=(selected_count >= 6 and not st.session_state.get(f"tool_{name}", False)))
        if checked and len(st.session_state.selected_tools) < 6:
            st.session_state.selected_tools.append(name)

image_url = "https://www.modelscope.cn/api/v1/studio/gongjy/MiniMind/repo?Revision=master&FilePath=images%2Flogo2.png&View=true"

st.markdown(
    f'<div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">'
    '<div style="font-style: italic; font-weight: 900; margin: 0; padding-top: 4px; display: flex; align-items: center; justify-content: center; flex-wrap: wrap; width: 100%;">'
    f'<img src="{image_url}" style="width: 40px; height: 40px; "> '
    f'<span style="font-size: 26px; margin-left: 10px;">{slogan}</span>'
    '</div>'
    f'<span style="color: #bbb; font-style: italic; margin-top: 6px; margin-bottom: 10px;">{get_text("disclaimer")}</span>'
    '</div>',
    unsafe_allow_html=True
)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _main():
    model, tokenizer = load_model_tokenizer(model_path)

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    messages = st.session_state.messages

    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{message["content"]}</div></div>',
                unsafe_allow_html=True)

    prompt = st.chat_input(key="input", placeholder=get_text('send'))

    if hasattr(st.session_state, 'regenerate') and st.session_state.regenerate:
        prompt = st.session_state.last_user_message
        regenerate_index = st.session_state.regenerate_index
        delattr(st.session_state, 'regenerate')
        delattr(st.session_state, 'last_user_message')
        delattr(st.session_state, 'regenerate_index')

    if prompt:
        display_before = copy.deepcopy(messages)
        context_before = copy.deepcopy(st.session_state.chat_messages)
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{prompt}</div></div>',
            unsafe_allow_html=True)
        placeholder = st.empty()

        random_seed = random.randint(0, 2 ** 32 - 1)
        setup_seed(random_seed)

        tools = [t for t in TOOLS if t['function']['name'] in st.session_state.get('selected_tools', [])] or None
        sys_prompt = [] if tools else [{"role": "system", "content": "你是MiniMind，一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题。"}]
        request_context = trim_context(context_before + [{"role": "user", "content": prompt}],
            st.session_state.history_chat_num, sys_prompt)
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if st.session_state.get('enable_thinking', False):
            template_kwargs["open_thinking"] = True
        if tools:
            template_kwargs["tools"] = tools
        if st.session_state.best_of_enabled:
            try:
                messages.append({"role": "user", "content": prompt})
                st.session_state.chat_messages = request_context
                with st.spinner('正在生成并挑选回答… / Generating and selecting…'):
                    result = best_of_n(model, tokenizer, st.session_state.chat_messages,
                        load_reward_selector(), n=st.session_state.candidate_count, seed=random_seed,
                        tools=tools, execute_tool=execute_tool,
                        temperature=st.session_state.temperature,
                        max_new_tokens=min(512, st.session_state.max_new_tokens),
                        thinking=False,
                        progress=lambda k, n: placeholder.info(f'生成候选 / Candidate {k+1}/{n}'))
                chosen = result['candidates'][result['selected']]
                display_parts = []
                for event in chosen['events']:
                    if event['role'] == 'assistant':
                        display_parts.append(event['content'])
                    else:
                        display_parts.append('工具结果 / Tool result (' + event['name'] + ')：\n```json\n' + event['content'] + '\n```')
                answer = '\n\n'.join(display_parts)
                placeholder.markdown(process_assistant_content(answer), unsafe_allow_html=True)
                st.caption(f"已选第 {result['selected']+1} / {len(result['candidates'])} 个回答 · Selected candidate")
                if result['fallback']:
                    st.warning('没有可评分的完整回答，保留第一个尝试；请缩短问题或关闭多候选。 / No complete scorable answer; showing the first attempt.')
                with st.expander('查看其他候选 / View candidates'):
                    for k, candidate in enumerate(result['candidates']):
                        st.markdown(f"**{k+1}** " + ('✓' if k == result['selected'] else ''))
                        st.write(candidate['final'])
                        if candidate['rm'] is not None:
                            st.caption(f"挑选分数 / Selection score: {candidate['rm']:.3f}")
                        else:
                            st.caption('未参与评分 / Not scored')
                messages.append({'role': 'assistant', 'content': answer})
                if result['fallback']:
                    # Keep the attempted response on screen, while excluding
                    # unfinished tool calls and failed prompts from inference.
                    st.session_state.chat_messages = context_before
                else:
                    st.session_state.chat_messages.extend(chosen['messages'])
            except BaseException as exc:
                messages[:] = display_before
                st.session_state.chat_messages = context_before
                if isinstance(exc, (RuntimeError, ValueError, OSError)):
                    placeholder.error('本次多候选未完成，请关闭开关后重试。 / Selection failed; turn off the option and retry.')
                    st.caption(str(exc))
                else:
                    raise
            return

        messages.append({"role": "user", "content": prompt})
        st.session_state.chat_messages = request_context
        new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)

        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "max_length": inputs.input_ids.shape[1] + st.session_state.max_new_tokens,
            "num_return_sequences": 1,
            "do_sample": True,
            "attention_mask": inputs.attention_mask,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "temperature": st.session_state.temperature,
            "top_p": 0.85,
            "streamer": streamer,
        }

        Thread(target=model.generate, kwargs=generation_kwargs).start()

        answer = ""
        for new_text in streamer:
            answer += new_text
            placeholder.markdown(process_assistant_content(answer, is_streaming=True), unsafe_allow_html=True)

        full_answer = answer
        for _ in range(16):
            tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', answer, re.DOTALL)
            if not tool_calls:
                break
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            tool_results = []
            for tc_str in tool_calls:
                try:
                    tc = json.loads(tc_str.strip())
                    result = execute_selected_tool(tc, tools, execute_tool)
                    st.session_state.chat_messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
                    tool_results.append(f'<div style="background: rgba(90, 130, 110, 0.20); border: 1px solid rgba(150, 200, 170, 0.30); padding: 10px 12px; border-radius: 12px; margin: 6px 0;"><div style="font-size:12px;opacity:.75;display:block;margin:0 0 6px 0;line-height:1;">ToolCalled</div><div><b>{tc.get("name", "")}</b>: {json.dumps(result, ensure_ascii=False)}</div></div>')
                except:
                    pass
            full_answer += "\n" + "\n".join(tool_results) + "\n"
            placeholder.markdown(process_assistant_content(full_answer, is_streaming=True), unsafe_allow_html=True)
            new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs["input_ids"] = inputs.input_ids
            generation_kwargs["attention_mask"] = inputs.attention_mask
            generation_kwargs["max_length"] = inputs.input_ids.shape[1] + st.session_state.max_new_tokens
            generation_kwargs["streamer"] = streamer
            Thread(target=model.generate, kwargs=generation_kwargs).start()
            answer = ""
            for new_text in streamer:
                answer += new_text
                placeholder.markdown(process_assistant_content(full_answer + answer, is_streaming=True), unsafe_allow_html=True)
            full_answer += answer
        answer = full_answer

        messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})


def main():
    with INFERENCE_LOCK:
        _main()


if __name__ == "__main__":
    main()
