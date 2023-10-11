import streamlit as st
from tools import gen_dat, gen_present_ind, ind_dat_edit
from datetime import date
import pandas as pd

st.set_page_config(layout="wide")


st.title("Bienvenue sur 📅 PLaNi")
# st.session_state
st.write("Veuillez spécifier la présence des collaborateurs :")

st.sidebar.title("📅 PLaNi")


def trip_cal(i):
    beg_pred_c()
    end_pred_c()
    if "collab_" + str(equipe[i]) in st.session_state:
        collab_c(i)


def all_cal(i):
    beg_pred_c()
    end_pred_c()
    if "absent_" + str(equipe[i]) in st.session_state:
        absent_c(i)
    if "collab_" + str(equipe[i]) in st.session_state:
        collab_c(i)


def beg_pred_c():
    st.session_state.beg_pred = st.session_state.beg_ent


if "beg_pred" not in st.session_state:
    st.sidebar.date_input(
        "Début des prédictions :", date.today(), key="beg_ent", on_change=beg_pred_c
    )
    st.session_state.beg_pred = st.session_state.beg_ent
else:  # without tis else, when beg_pred is initialised in the ses_stt, the if above is
    # not read so the date_input disappear, so this else is useful and I think also for every ses_stt with callb
    st.sidebar.date_input(
        "Début des prédictions :",
        st.session_state.beg_pred,
        key="beg_ent",
        on_change=beg_pred_c,
    )


def end_pred_c():
    st.session_state.end_pred = st.session_state.end_ent


if "end_pred" not in st.session_state:
    st.sidebar.date_input(
        "Fin des prédictions :",
        date.fromisocalendar(
            date.today().isocalendar()[0], st.session_state.beg_pred.isocalendar()[1], 5
        ),
        key="end_ent",
        on_change=end_pred_c,
    )
    st.session_state.end_pred = st.session_state.end_ent
else:
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


def absent_c(i):
    st.session_state["absent_" + str(equipe[i])] = st.session_state[
        "ent_absent_" + str(equipe[i])
    ]


def collab_c(i):
    st.session_state["collab_" + str(equipe[i])] = gen_present_ind(
        st.session_state.beg_pred,
        st.session_state.end_pred,
        '30T',
        collabs=df_collabs.loc[:, equipe[i]].loc[
            :,
            [
                collab
                for collab in list(df_collabs.loc[:, equipe[i]].columns)
                if collab not in set(st.session_state["absent_" + str(equipe[i])])
            ],
        ],
        dict_=st.session_state["ent_collab_" + str(equipe[i])],
    )


for i in range(len(tabs)):
    with tabs[i]:
        if "absent_" + str(equipe[i]) not in st.session_state:
            st.multiselect(
                "Qui est absent : ",
                options=df_collabs.loc[:, equipe[i]].columns,
                key="ent_absent_" + str(equipe[i]),
                on_change=all_cal,
                args=(i,),
            )
            st.session_state["absent_" + str(equipe[i])] = st.session_state[
                "ent_absent_" + str(equipe[i])
            ]  # this session_state stocks the modified dfs for presence
        else:
            st.multiselect(
                "Qui est absent : ",
                options=df_collabs.loc[:, equipe[i]].columns,
                default=st.session_state["absent_" + str(equipe[i])],
                key="ent_absent_" + str(equipe[i]),
                on_change=all_cal,
                args=(i,),
            )

        if "collab_" + str(equipe[i]) not in st.session_state:
            st.session_state["collab_" + str(equipe[i])] = gen_present_ind(
                st.session_state.beg_pred,
                st.session_state.end_pred,
                '30T',
                df_collabs.loc[:, equipe[i]].loc[
                    :,
                    [
                        collab
                        for collab in list(df_collabs.loc[:, equipe[i]].columns)
                        if collab
                        not in set(st.session_state["absent_" + str(equipe[i])])
                    ],
                ],
            )
            st.data_editor(
                ind_dat_edit(st.session_state["collab_" + str(equipe[i])]),
                key="ent_collab_" + str(equipe[i]),
                on_change=all_cal,
                args=(i,),
            )
        else:
            st.data_editor(
                ind_dat_edit(
                    gen_present_ind(
                        st.session_state.beg_pred,
                        st.session_state.end_pred,
                        '30T',
                        df_collabs.loc[:, equipe[i]].loc[
                            :,
                            [
                                collab
                                for collab in list(df_collabs.loc[:, equipe[i]].columns)
                                if collab
                                not in set(st.session_state["absent_" + str(equipe[i])])
                            ],
                        ],
                    )
                ),
                key="ent_collab_" + str(equipe[i]),
                on_change=all_cal,
                args=(i,),
            )  # à l'avenir : il est mieux de mettre le df modifié stored dans le session_state ici
# pour garder une trace visuelle des modifications pour le user
