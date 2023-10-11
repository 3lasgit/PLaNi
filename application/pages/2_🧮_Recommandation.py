import streamlit as st
import joblib
import pandas as pd
from tools_new import (
    gen_dat,
    gen_present_ind,
    unr_mod,
    col_theor,
    ind_dat_edit,
    new_solve_planning_problem,
    filler_team,
)
import plotly.io as pio

# st.session_state
pio.templates.default = "plotly"

lgbm_30T = joblib.load(
        "./application/models/lgbm_reg_H.pkl"
    )  # start thinking of st.cachedata


st.title("Recommandations par heure")

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
df_collabs.fillna("2.5", inplace=True)
df_collabs.loc["Appel/heure"] = df_collabs.loc["Appel/heure"].apply(
    lambda s: float(str(s).replace(",", "."))
)
df_collabs.sort_values(
    by=["equipe", "Appel/heure"], axis=1, ascending=False, inplace=True
)
df_collabs.loc["Mail/heure"] = 12  # demander au CLC de compléter les stats : Valérie

equipe = list(df_collabs.columns.get_level_values(0).unique())

m_a, m, m_m = 6, 15, 12
n_col = 0
for i in range(4):
    n_col += len(
        gen_present_ind(
            st.session_state.beg_pred,
            st.session_state.end_pred,
            '30T',
            df_collabs.loc[:, equipe[i]].loc[
                :,
                [
                    collab
                    for collab in list(df_collabs.loc[:, equipe[i]].columns)
                    if collab not in set(st.session_state["absent_" + str(equipe[i])])
                ],
            ],
        ).columns
    )


tab1, tab2, tab3, tab4 = st.tabs(equipe)
tabs = [tab1, tab2, tab3, tab4]


def planning_c(i):
    st.session_state["planning_" + str(equipe[i])] = unr_mod(
        st.session_state["ent_planning_" + str(equipe[i])],
        st.session_state["planning_" + str(equipe[i])],
    )
    st.session_state["planning_" + str(equipe[i])]["§Appel"] = col_theor(
        st.session_state["planning_" + str(equipe[i])],
        df_collabs.loc[:, equipe[i]].loc[
            :,
            [
                collab
                for collab in list(df_collabs.loc[:, equipe[i]].columns)
                if collab not in set(st.session_state["absent_" + str(equipe[i])])
            ],
        ],
        "Appel",
    )
    st.session_state["planning_" + str(equipe[i])]["§Mail"] = col_theor(
        st.session_state["planning_" + str(equipe[i])],
        df_collabs.loc[:, equipe[i]].loc[
            :,
            [
                collab
                for collab in list(df_collabs.loc[:, equipe[i]].columns)
                if collab not in set(st.session_state["absent_" + str(equipe[i])])
            ],
        ],
        "Mail",
    )
    # st.session_state["planning_" + str(equipe[i])]["QS Appel"] = (
    #     st.session_state["planning_" + str(equipe[i])]["§Appel"] * 100 / st.session_state[
    #         "planning_" + str(equipe[i])]["#Appel"]
    # ).astype("int")
    # st.session_state["planning_" + str(equipe[i])]["QS Mail"] = (
    #     st.session_state["planning_" + str(equipe[i])]["§Mail"] * 100 / st.session_state[
    #         "planning_" + str(equipe[i])]["Mail à traiter"]
    # ).astype("int")


for i in range(len(tabs)):
    with tabs[i]:
        e_col = len(
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
            ).columns
        )
        p_col = e_col / n_col
        if "planning_" + str(equipe[i]) not in st.session_state:
            # Attention le code en-dessous ne se met pas à jour lorsque l'on modifie les entrées en première page
            # (cf le callback qui ne remeonte pas à la première page c'est normal), il faut définir dans le scénario
            # que si l'utilisateur a besoin de revenir en page 1 il faut recharger toutes les pages et recommencer
            st.session_state["planning_" + str(equipe[i])] = filler_team(
                gen_dat(st.session_state.beg_pred, st.session_state.end_pred, '30T'),
                st.session_state[
                    "collab_" + str(equipe[i])
                ],
                # attention ici si entre les changements de page,
                # il y a un changement dans les sess_stat de date, trouver une solution pour
                # tirer l'info depuis la sess_stat du df directement pour bien avoir la date qui
                # correspond bien avec celle du df
                df_collabs.loc[:, equipe[i]].loc[
                    :,
                    [
                        collab
                        for collab in list(df_collabs.loc[:, equipe[i]].columns)
                        if collab
                        not in set(st.session_state["absent_" + str(equipe[i])])
                    ],
                ],
                lgbm_30T,
                m_a,
                m,
                m_m,
                e_col,
                p_col,
            )
            st.data_editor(
                ind_dat_edit(st.session_state["planning_" + str(equipe[i])]),
                key="ent_planning_" + str(equipe[i]),
                on_change=planning_c,
                args=(i,),
                column_config=dict.fromkeys(
                    df_collabs.loc[:, equipe[i]]
                    .loc[
                        :,
                        [
                            pers
                            for pers in list(df_collabs.loc[:, equipe[i]].columns)
                            if pers
                            not in set(st.session_state["absent_" + str(equipe[i])])
                        ],
                    ]
                    .columns,
                    st.column_config.SelectboxColumn(
                        help="Attribuez les rôles",
                        required=True,
                        width="small",
                        options=["📞 Appel", "📧 Mail"],
                    ),
                ),
            )
        else:
            st.data_editor(
                ind_dat_edit(st.session_state["planning_" + str(equipe[i])]),
                key="ent_planning_" + str(equipe[i]),
                on_change=planning_c,
                args=(i,),
                column_config=dict.fromkeys(
                    df_collabs.loc[:, equipe[i]]
                    .loc[
                        :,
                        [
                            pers
                            for pers in list(df_collabs.loc[:, equipe[i]].columns)
                            if pers
                            not in set(st.session_state["absent_" + str(equipe[i])])
                        ],
                    ]
                    .columns,
                    st.column_config.SelectboxColumn(
                        help="Attribuez les rôles",
                        required=True,
                        width="small",
                        options=["📞 Appel", "📧 Mail"],
                    ),
                ),
            )
            # à l'avenir : il est mieux de mettre le df modifié stored dans le session_state
            # ici pour garder une trace visuelle des modifications pour le user

        st.write(
            new_solve_planning_problem(
                10,
                12,
                (st.session_state["collab_" + str(equipe[i])]).iloc[0],
                df_collabs.loc[:, equipe[i]],
            )
        )
        # df_collabs.loc[:,equipe[i]].loc[
        # :,st.session_state['collab_'+str(equipe[i])].iloc[0].loc[
        # st.session_state['collab_'+str(equipe[i])].iloc[0]==True].index]
