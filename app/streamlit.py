import streamlit as st
from scrapper.ai_query import QueryAgent




st.title("💬RAGFin test")

# Lưu trữ hội thoại
QueryAgent = QueryAgent()
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "Bạn là một trợ lý hữu ích."}
    ]

# Hiển thị các tin nhắn đã gửi
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ô nhập liệu cho người dùng
if prompt := st.chat_input("Nhập tin nhắn của bạn..."):
    # Hiển thị tin nhắn người dùng
    with st.chat_message("user"):
        st.markdown(prompt)

    # Thêm tin nhắn người dùng vào session
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    reply = QueryAgent.query(prompt)

    # Hiển thị phản hồi
    with st.chat_message("assistant"):
        st.markdown(reply)

    # Thêm phản hồi vào session
    st.session_state.messages.append({"role": "assistant", "content": reply})