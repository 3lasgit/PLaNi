from pathlib import Path

import streamlit as st
from st_pages import Page, show_pages

FILE_PATH = Path(__file__).resolve().parent

st.set_page_config(
    page_title="PLaNi",
    page_icon="📈",
)
st.title("Bienvenue sur 📅 PLaNi")
st.write("Veuillez cliquer sur 🙋‍♂️ Présence pour commercer.")
show_pages(
    [
        Page(
            f"{FILE_PATH}/pages/1_🙋‍♂️_Présence.py",
            "Présence",
        ),
        Page(
            f"{FILE_PATH}/pages/2_🧮_Recommandation.py",
            "Recommandation",
        ),
        Page(
            f"{FILE_PATH}/pages/3_📞📧_Résultat.py",
            "Résultat",
        ),
    ]
)
