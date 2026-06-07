import os
import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain.agents import create_agent 
from langchain.agents.middleware import wrap_tool_call

from tools import get_weather, get_news

load_dotenv()

st.set_page_config(page_title="City Intelligence Agent", page_icon="🏙️", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    h1 { font-family: 'Inter', sans-serif; font-weight: 700; color: #ffffff !important; letter-spacing: -0.5px; }
    .chat-bubble { padding: 16px 20px; border-radius: 16px; margin-bottom: 12px; line-height: 1.5; font-size: 15px; max-width: 85%; }
    .user-bubble { background: linear-gradient(135deg, #1f293d, #2d3748); color: #f7fafc; margin-left: auto; border: 1px solid #3e4c66; }
    .bot-bubble { background-color: #161b22; color: #e6edf3; margin-right: auto; border: 1px solid #21262d; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

if "llm" not in st.session_state:
    st.session_state.llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.2,
        api_key=os.getenv("MISTRAL_API_KEY"),
        mistral_api_key=os.getenv("MISTRAL_API_KEY")
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_tool_call" not in st.session_state:
    st.session_state.pending_tool_call = None

if "agent_pipeline" not in st.session_state:
    
    @wrap_tool_call
    def web_human_approval(request, handler):
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call["args"]
        
        st.session_state.pending_tool_call = {
            "request": request,
            "handler": handler,
            "name": tool_name,
            "args": tool_args
        }
        return ToolMessage(content="WAIT_FOR_HUMAN_APPROVAL", tool_call_id=request.tool_call["id"])

    st.session_state.agent_pipeline = create_agent(
        st.session_state.llm,
        tools=[get_weather, get_news],
        system_prompt="You are a premium, highly polite city assistant. Use tools dynamically when live status/metrics are queried.",
        middleware=[web_human_approval]
    )

@st.dialog("🛡️ Autonomous Agent Authorization Required")
def show_approval_popup():
    call_info = st.session_state.pending_tool_call
    st.warning(f"The Autonomous Agent is requesting local system permissions to execute an external tool node.")
    st.markdown(f"**🛠️ Target Tool Node:** `{call_info['name']}`")
    st.json(call_info['args'])
    st.markdown("Do you authorize this system execution?")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve & Execute", use_container_width=True, type="primary"):
            real_response = call_info["handler"](call_info["request"])
            
            # 🎯 ટૂલનો અસલી આન્સર હિસ્ટ્રીમાં સેવ કરો
            st.session_state.chat_history.append({"role": "bot", "content": real_response.content})
            st.session_state.pending_tool_call = None
            st.rerun()
    with col2:
        if st.button("❌ Deny Call", use_container_width=True):
            st.session_state.chat_history.append({"role": "bot", "content": f"⚠️ Execution of `{call_info['name']}` was blocked by the user."})
            st.session_state.pending_tool_call = None
            st.rerun()

st.title("🏙️ City Intelligence Agent")
st.caption("A premium LangChain Agent with Human-in-the-Loop Governance & Mistral AI Core.")
st.markdown("---")

# UI પર જૂની ચેટ રેન્ડર કરવી
chat_container = st.container()
with chat_container:
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f'<div class="chat-bubble user-bubble"><b>You:</b><br>{chat["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble bot-bubble"><b>Bot:</b><br>{chat["content"]}</div>', unsafe_allow_html=True)

user_query = st.chat_input("Ask about weather, news or general information...")

if user_query:
    # ૧. યુઝરના સવાલને લિસ્ટમાં એડ કરો
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.rerun()

# 🎯 મોસ્ટ ઈમ્પોર્ટન્ટ ફિક્સ: જો છેલ્લો મેસેજ યુઝરનો હોય, તો જ બોટ પાસે પ્રોસેસ કરાવવું
if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
    with st.spinner("Agent thinking..."):
        try:
            # 🧠 આખી ચેટ હિસ્ટ્રીને LangChain ના મેસેજ ઓબ્જેક્ટ્સમાં કન્વર્ટ કરો (મેમરી લૂપ)
            formatted_messages = []
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    formatted_messages.append(HumanMessage(content=msg["content"]))
                else:
                    formatted_messages.append(AIMessage(content=msg["content"]))
            
            # આખો ઇતિહાસ (Memory) મોડેલને પાસ કરો
            result = st.session_state.agent_pipeline.invoke({
                "messages": formatted_messages
            })
            
            bot_reply = result['messages'][-1].content
            
            if "WAIT_FOR_HUMAN_APPROVAL" not in bot_reply:
                st.session_state.chat_history.append({"role": "bot", "content": bot_reply})
                st.rerun()
                
        except Exception as err:
            st.error(f"Execution Interrupted: {err}")

if st.session_state.pending_tool_call is not None:
    show_approval_popup()