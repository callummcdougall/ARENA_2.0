import streamlit as st
import base64
import platform
from streamlit_chat import message
import openai
import json
import plotly.io as pio
import re
# from st_on_hover_tabs import on_hover_tabs

is_local = (platform.processor() != "")

def read_from_html(filename):
    with open(filename) as f:
        html = f.read()
    call_arg_str = re.findall(r'Plotly\.newPlot\((.*)\)', html)[0]
    call_args = json.loads(f'[{call_arg_str}]')
    try:
        plotly_json = {'data': call_args[1], 'layout': call_args[2]}
        fig = pio.from_json(json.dumps(plotly_json))
    except:
        del call_args[2]["template"]["data"]["scatter"][0]["fillpattern"]
        plotly_json = {'data': call_args[1], 'layout': call_args[2]}
        fig = pio.from_json(json.dumps(plotly_json))
    return fig

def on_hover(title, content):
        st.markdown(f"""
<div class="myDIV">{title}</div>
<div class="hide">
{content}
</div>
""", unsafe_allow_html=True)

def st_image(name, width, return_html=False):
    with open("media/" + name, "rb") as file:
        img_bytes = file.read()
    encoded = base64.b64encode(img_bytes).decode()
    img_html = f"<img style='width:{width}px;max-width:100%;margin-bottom:25px' src='data:image/png;base64,{encoded}' class='img-fluid'>"
    if return_html:
        return img_html
    st.markdown(img_html, unsafe_allow_html=True)

def st_excalidraw(name, width):
    img_html_full = ""
    for suffix in ["light", "dark"]:
        with open("images/" + name + "-" + suffix + ".png", "rb") as file:
            img_bytes = file.read()
        encoded = base64.b64encode(img_bytes).decode()
        img_html = f"<img style='width:{width}px;max-width:100%;margin-bottom:25px' class='img-fluid {suffix}Excalidraw' src='data:image/png;base64,{encoded}'>"
        img_html_full += img_html
    st.markdown(img_html_full, unsafe_allow_html=True)

def styling():
    st.set_page_config(layout="wide", page_icon="🔬")
    st.markdown(r"""
<style>
img {
    margin-bottom: 15px;
    max-width: 100%;
}
.myDIV {
    margin-bottom: 15px;
}
.hide {
    display: none;
}
.myDIV:hover + .hide {
    display: block;
    float: left;
    position: absolute;
    z-index: 1;
}
.stAlert h4 {
    padding-top: 0px;
}
.st-ae code {
    padding: 0px !important;
}
label.effi0qh3 {
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 15px;
}
p {
    line-height:1.48em;
}
.st-ae h2 {
    margin-top: -15px;
}
.streamlit-expanderHeader {
    font-size: 1em;
    color: darkblue;
}
.css-ffhzg2 .streamlit-expanderHeader {
    color: lightblue;
}
header {
    background: rgba(255, 255, 255, 0) !important;
}
code:not(pre code) {
    color: red !important;
}
pre code {
    white-space: pre-wrap !important;
    font-size:13px !important;
}
.st-ae code {
    padding: 4px;
}
.css-ffhzg2 .st-ae code: not(stCodeBlock) {
    background-color: black;
}
code:not(h1 code):not(h2 code):not(h3 code):not(h4 code) {
    font-size: 13px;
}
a.contents-el > code {
    color: black;
    background-color: rgb(248, 249, 251);
}
.css-ffhzg2 a.contents-el > code {
    color: orange !important;
    background-color: rgb(26, 28, 36);
}
.css-ffhzg2 code:not(pre code) {
    color: orange !important;
}
.css-ffhzg2 .contents-el {
    color: white !important;
}
.css-fg4pbf blockquote {
    background-color: rgb(231,242,252);
    padding: 15px 20px 5px;
    border-left: 0px solid rgb(230, 234, 241);
}
.katex {
    font-size:18px;
}
h2 .katex, h3 .katex, h4 .katex {
    font-size: unset;
}
ul {
    margin-bottom: 15px;
}
ul.contents {
    line-height:1.3em; 
    list-style:none;
    color-black;
    margin-left: -15px;
}
li.margtop {
    margin-top: 10px !important;
}
ul.contents a, ul.contents a:link, ul.contents a:visited, ul.contents a:active {
    color: black;
    text-decoration: none;
}
ul.contents a:hover {
    color: black;
    text-decoration: underline;
}
details {
    margin-bottom: 10px;
    padding: 5px 15px 1px;
    background-color: #eee;
    border-radius: 10px;
}
summary {
    margin-bottom: 5px;
}
.css-fg4pbf pre {
    background: rgb(247, 248, 250);
}
.css-fg4pbf code {
    background: rgb(247, 248, 250);
}
</style>""", unsafe_allow_html=True)
    
#     div[data-testid="column"] {
#     background-color: #f9f5ff;
#     padding: 15px;
# }

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True



def concat_lists(lists):
    return [item for sublist in lists for item in sublist]

def parse_text_from_page(page_name, func_name):
    filename = "Home.py" if page_name=="home" else f"pages/{page_name}.py"
    with open(filename, encoding="utf-8") as file:
        lines = file.readlines()
    s = ""
    prepending = False
    correct_page = False
    last_was_start_or_end = "end"
    for line in lines:
        if line.strip() == "# start":
            prepending = True
            assert last_was_start_or_end == "end"
            last_was_start_or_end = "start"
        elif line.strip() == "# end":
            assert last_was_start_or_end == "start"
            last_was_start_or_end = "end"
            prepending = False
        elif line.strip() == f"def {func_name}():":
            correct_page = True
        elif re.match(r"def [a-z_]+\(\):", line.strip()):
            correct_page = False
        elif all([
            prepending,
            correct_page,
            not re.search(r"st\.[a-z]+\(", line.strip()),
            not line.strip().startswith(r"st_image"),
            line.strip() not in ['""")', "''')"]
        ]):
            s += line

        # some checks for me!
        if any([line.strip().endswith(j) for j in ['""")', "''')"]]) and not line.startswith("# "):
            assert len(line.strip()) == 4, f"Need to add line breaks at the end of text. Failed with line: {line}"
    return s



def get_tokenized_len(text, tokenizer):
    total = 0
    while len(text) > 0:
        text_next, text = text[:tokenizer.model_max_length], text[tokenizer.model_max_length:]
        total += len(tokenizer.encode(text_next))
    return total



def chatbot_setup(
    prepend: str = "", 
    debug: bool = False,
    run_anyway: bool = True,
    new_section: bool = False,
    new_page: bool = False,
):

    try:
        openai.api_key = st.secrets["api_secret"]
    except:
        st.sidebar.error("Please set the OpenAI API key. See the homepage for instructions.")
        return

    # Storing the chat
    for k, v in zip(['generated', 'past', 'prepend'], [[], [], prepend]):
        if k not in st.session_state or (k == 'prepend' and (new_page or new_section)):
            st.session_state[k] = v

    # We will get the user's input by calling the get_text function
    def get_text():
        extra_context = st.text_area("Prompt:", "", key="input", placeholder="Type your prompt here, then press Ctrl+Enter.\nThe prompt will be prepended with most of the page content (so you can ask questions about the material).")
        # input_text = st.text_input("Prompt:", "", key="input")
        return extra_context #, input_text

    def generate_response(prompt):
        '''
        This function will generate a response from the prompt

        The openai api has different syntax for different types of 
        models, so we need to split depending on whether it is a
        ChatCompletion or Completion model.
        '''
        if chatbot in ["gpt-3.5-turbo", "gpt-4"]:
            chat_history = []
            for user_input, output in zip(st.session_state["past"], st.session_state["generated"]):
                chat_history.extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": output}
                ])
            if debug:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                chat_history_raw = "\n".join([d["content"] for d in chat_history])
                st.sidebar.write("\n\n".join([
                    f"Prepend: {get_tokenized_len(st.session_state['prepend'], tokenizer)}",
                    f"Chat history (raw): {get_tokenized_len(chat_history_raw, tokenizer)}",
                    f"Prompt: {get_tokenized_len(prompt, tokenizer)}",
                ]))
                st.write(st.session_state['prepend'])
            if (not debug) or run_anyway:
                completions = openai.ChatCompletion.create(
                    model = chatbot,
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "assistant", "content": st.session_state["prepend"]},
                        *chat_history,
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens = 512,
                )
                return completions.choices[0].message.content
        elif chatbot in ["text-davinci-003"]:
            chat_history = ""
            for user_input, output in zip(st.session_state["past"], st.session_state["generated"]):
                chat_history += "\n\n" + user_input + "\n\n" + output
            if debug:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                st.sidebar.write("\n".join([
                    f"Prepend: {get_tokenized_len(st.session_state['prepend'], tokenizer)}",
                    f"Chat history: {get_tokenized_len(chat_history, tokenizer)}",
                    f"Prompt: {get_tokenized_len(prompt, tokenizer)}",
                ]))
                st.write(st.session_state['prepend'])
            if (not debug) or run_anyway:
                completions = openai.Completion.create(
                    engine = chatbot,
                    prompt = st.session_state["prepend"] + "\n\n" + chat_history + "\n\n" + prompt,
                    max_tokens = 512,
                    n = 1,
                    stop = None,
                    temperature=0.5,
                )
                return completions.choices[0].text
        else:
            raise ValueError(f"Unknown chatbot: {chatbot}")

    with st.sidebar:
        st.markdown("---")
        chatbot = st.selectbox(
            "Select a chatbot",
            ["text-davinci-003", "gpt-3.5-turbo", "gpt-4"],
            index=1
        )
        st.button(
            "Clear chat",
            key="clear",
            on_click=lambda: st.session_state.clear(),
        )
        user_input = get_text()
        if user_input and not new_section and not new_page:
            output = generate_response(user_input).strip()
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
        if st.session_state['generated']:
            for i, (chatbot_msg, user_msg) in reversed(list(enumerate(zip(
                st.session_state['generated'], 
                st.session_state['past']
            )))):
                message(chatbot_msg, key=str(i))
                message(user_msg, is_user=True, key=str(i) + '_user')
        st.markdown("")
