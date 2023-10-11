import streamlit as st
import joblib
import plotly.graph_objects as go
from tools import gen_dat, color_background
import pandas as pd
import plotly.io as pio


pio.templates.default = "plotly"

lgbm_30T = joblib.load(
        "./application/models/lgbm_reg_H.pkl"
    )


st.title("Planning final")

st.sidebar.title("Vos choix sont :")


# Date de début
def beg_pred_c():
    st.session_state.beg_pred = st.session_state.beg_ent


st.sidebar.date_input(
    "Début des prédictions :",
    st.session_state.beg_pred,
    key="beg_ent",
    on_change=beg_pred_c,
)


# Date de fin
def end_pred_c():
    st.session_state.end_pred = st.session_state.end_ent


# st.session_state.end_pred = st.sidebar.date_input("Fin des prédictions :", st.session_state.end_pred); non fonctionnel
st.sidebar.date_input(
    "Fin des prédictions :",
    st.session_state.end_pred,
    key="end_ent",
    on_change=end_pred_c,
)

df_collabs = pd.read_csv(
    "./application/data/stats_simplifiées_collab.csv",
    delimiter=",",
    header=[0, 1],
    index_col=0,
)  # pourrait être un fichier rempli par l'utilisateur N.1 qu'il entre à chaque fois qu'il va faire un planning
df_collabs.fillna("-1,0", inplace=True)
df_collabs.loc["Appel/heure"] = df_collabs.loc["Appel/heure"].apply(
    lambda s: float(str(s).replace(",", "."))
)
df_collabs.sort_values(
    by=["equipe", "Appel/heure"], axis=1, ascending=False, inplace=True
)

dat = gen_dat(st.session_state.beg_pred, st.session_state.end_pred, '30T')

equipe = list(df_collabs.columns.get_level_values(0).unique())

tab1, tab2, tab3, tab4 = st.tabs(equipe)
tabs = [tab1, tab2, tab3, tab4]

for i in range(len(tabs)):
    with tabs[i]:
        col1, col2 = st.columns(2)
        with col1:
            pred = lgbm_30T.predict(dat).astype(int)
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    name="#Appel",
                    x=dat.index,
                    y=st.session_state["planning_" + str(equipe[i])].loc[:, "#Appel"],
                )
            )  # , mode='lines'))
            fig.add_trace(
                go.Bar(
                    name="Mail à traiter",
                    x=dat.index,
                    y=st.session_state["planning_" + str(equipe[i])].loc[
                        :, "Mail à traiter"
                    ],
                )
            )  # , mode='lines'))

            fig.update_layout(
                template="plotly",
                title="Prédiction des appels",  # xaxis_title='',
                yaxis_title="Volume",
                plot_bgcolor="lightblue",  # Changer la couleur de fond
                title_x=0.45,
                barmode="stack",
            )
            st.plotly_chart(fig)

        with col2:
            planning_styled = st.session_state[
                "planning_" + str(equipe[i])
            ].style.applymap(color_background)
            st.dataframe(planning_styled)
