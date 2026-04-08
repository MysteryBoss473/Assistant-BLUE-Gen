import streamlit as st

st.title("Ma première app Streamlit")
st.write("Bonjour 👋")

nom = st.text_input("Entrez votre nom")
if nom:
    st.write(f"Salut {nom} !")