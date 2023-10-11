import pandas as pd
from datetime import timedelta
import datetime
from vacances_scolaires_france import SchoolHolidayDates
from pulp import LpProblem, LpMaximize, LpVariable, PULP_CBC_CMD, LpStatus, value
from pandas.api.types import CategoricalDtype
from workalendar.europe import France

d = SchoolHolidayDates()


cal = France()


# Génère un df à entrer dans lgbm pour des preds
def gen_dat(beg_pred, end_pred, freq):
    plage = pd.date_range(start=beg_pred, end=end_pred + timedelta(1), freq=freq)
    # timedelta(1) to include the end in the generation,
    # in fact it generates to the end but only at midnight so it is dropped afterward
    # freq est la fréquence à laquelle on agreg : 1H,30T,2H
    df = pd.DataFrame(index=plage)
    df["Consideration Start"] = df.index
    # Ajout des features temporelles détaillées
    df = df.assign(
        year=df["Consideration Start"].dt.isocalendar()["year"],
        week=df["Consideration Start"].dt.isocalendar()["week"],
        day=df["Consideration Start"].dt.isocalendar()[
            "day"
        ],  # quarter=df['Consideration Start'].dt.quarter,
        month=df["Consideration Start"].dt.month,
        day_mon=df["Consideration Start"].dt.day,
        hour=df["Consideration Start"].dt.hour * 100 + df[
            "Consideration Start"].dt.minute,
    )

    # Éjection des H off-office lorsque l'on est freq=min
    df.drop(index=df.loc[(df["hour"] < 800) | (df["hour"] >= 1800)].index, inplace=True)

    # Conversion de la colonne de date en index
    df.set_index("Consideration Start", inplace=True)

    # Éjection des samedis et dimanches lorsque l'on est en phone only
    df.drop(index=df.loc[(df["day"] == 6) | (df["day"] == 7)].index, inplace=True)

    # Conversion dans le type adéquat pour LGBM
    df = df.astype(int)

    # Rajoute les vacances, jours fériés et d-1 fériés
    df = df.assign(
        vac_A=df.apply(
            lambda x: d.is_holiday_for_zone(
                datetime.date(x["year"], x["month"], x["day_mon"]), "A"
            ),
            axis=1,
        ),
        vac_B=df.apply(
            lambda x: d.is_holiday_for_zone(
                datetime.date(x["year"], x["month"], x["day_mon"]), "B"
            ),
            axis=1,
        ),
        vac_C=df.apply(
            lambda x: d.is_holiday_for_zone(
                datetime.date(x["year"], x["month"], x["day_mon"]), "C"
            ),
            axis=1,
        ),
        ouvre=df.apply(
            lambda x: cal.is_working_day(
                datetime.date(x["year"], x["month"], x["day_mon"])
            ),
            axis=1,
        ),
        begin=df.apply(lambda x: 800 <= x["hour"] < 900, axis=1),
        lunch=df.apply(lambda x: 1200 <= x["hour"] < 1300, axis=1),
        end=df.apply(lambda x: 1700 <= x["hour"] < 1800, axis=1),
    )

    df["d-1_fer"] = ((df["ouvre"].shift(1, fill_value=True) - 1) * -1).apply(bool)
    df.loc[(df["day"] == 1) & (df["d-1_fer"]), "d-1_fer"] = False

    # Éjection des samedis et dimanches lorsque l'on est en phone only ainsi que les jours fériés
    df.drop(index=df.loc[~df.ouvre].index, inplace=True)  # ~ permet de donner directement les FALSE
    # et d'éviter les comparaisons == FALSE et donc les erreurs E712
    df.drop(columns="ouvre", inplace=True)
    return df


# Optimiser


def solve_planning_problem(a, m_a, m, m_m, n_col, printer=False):
    # Création du problème
    prob = LpProblem("Planning Problem", LpMaximize)

    # Variables de décision
    s_a = LpVariable("s_a", cat="Integer")
    s_m = LpVariable("s_m", cat="Integer")

    # Fonction objectif
    prob += s_a * m_a + s_m * m_m

    # Contraintes
    prob += s_a + s_m <= n_col  # Contrainte de ressources dispo
    prob += s_a * m_a >= a  # Contrainte de productivité phone
    prob += s_m * m_m >= m  # Contrainte de productivité mail

    prob += s_a >= 0  # Contrainte de non-négativité
    prob += s_m >= 0  # Contrainte de non-négativité

    # Résolution du problème
    prob.solve(PULP_CBC_CMD(msg=0))

    if printer:  # unused and not authorised by flake
        None  # Affichage du statut de la solution
        None  # print("Statut:", LpStatus[prob.status])
        None  # Affichage de la valeur optimale des variables
        None  # print("Valeur de s_a:", value(s_a))
        None  # print("Valeur de s_m:", value(s_m))
        None  # Affichage de la valeur optimale de la fonction objectif
        None  # print("Valeur optimale de la fonction objectif:", value(prob.objective))
    return value(s_a), value(s_m), LpStatus[prob.status], value(prob.objective)


# Présence


def ind_dat_edit(df):
    if df.index.dtype == "datetime64[ns]":
        df.index = df.index.strftime("%Y-%m-%d %H:%M")
    return df


def old_gen_present(dat, df_collabs):
    collab_pres = pd.DataFrame(True, index=dat.index, columns=df_collabs.columns)
    return collab_pres


def old_gen_present_ind_2(
    dat, df_collabs
):  # same as gen_present but formats directly the index in order to work with st.data_editor
    collab_pres = pd.DataFrame(True, index=dat.index, columns=df_collabs.columns)
    collab_pres.index = collab_pres.index.strftime("%Y-%m-%d %H:%M")
    return collab_pres


def unr_mod(dict_, base):
    edited_rows = dict_["edited_rows"]
    index_ = base.index
    # Unraveller and modifier for sessions_state stemmed from a data_editor dict
    for i in edited_rows:
        ind = index_[i]
        base.loc[ind, edited_rows[i].keys()] = edited_rows[i].values()
    return base


def gen_present_ind(
    beg_pred, end_pred, freq, collabs, dict_=None
):  # same as gen_present but formats directly the index in order to work with st.data_editor
    dat = gen_dat(beg_pred, end_pred, freq)
    if dict_ is not None:
        collab_pres = pd.DataFrame(True, index=dat.index, columns=collabs.columns)
        collab_pres = unr_mod(dict_, ind_dat_edit(collab_pres))
    else:
        collab_pres = pd.DataFrame(True, index=dat.index, columns=collabs.columns)
    return collab_pres


# pour garder les modifications des cases malgré le changement, essayer ça dans mod... :
# - prendre en compte tous les cas de figures issus des changements de dates en les transcrivant
# en structure if, les changements de personnels étant déjà pris en compte par le session_state et
# avec le bon callback ça devrait faire l'affaire

# NON UTILISÉE POUR LE MOMENT

def empty_df():  # évite l'erreur B008 Do not perform function calls in argument defaults de flake
    return pd.DataFrame()


def mod_present(
    beg_pred, end_pred, freq, collabs, base=None, dict_=None
):  # same as gen_present but formats directly the index in order to work with st.data_editor
    dat = gen_dat(beg_pred, end_pred, freq)
    if base is None:  # base=pd.DataFrame(True,index=base.index, columns=base.columns)
        # # initialisation de collab_pres qui pose problème : dès qu'il y a un
        # changement rerun tout ça et donc remets tout en True
        if (
            beg_pred < datetime.datetime.strptime(base.index[0], "%Y-%m-%d %H:%M").date()
        ):  # transformer en datetime
            index_haut = gen_dat(
                beg_pred,
                datetime.datetime.strptime(base.index[0], "%Y-%m-%d %H:%M").date(),
                freq,
            ).index
            complet = pd.DataFrame(True, index=index_haut, columns=base.columns)
            base = pd.concat([complet, base])
        if (
            end_pred > datetime.datetime.strptime(base.index[-1], "%Y-%m-%d %H:%M").date()
        ):
            index_bas = gen_dat(
                datetime.datetime.strptime(base.index[-1], "%Y-%m-%d %H:%M").date(),
                end_pred,
                freq,
            ).index
            complet = pd.DataFrame(True, index=index_bas, columns=base.columns)
            base = pd.concat([base, complet])
        # if collabs!=base.columns : # il faut peut être les mettre en set
        for column in collabs.columns:
            if column not in base:
                base[str(column)] = True
        edited_rows = dict_["edited_rows"]
        index_ = dat.index.strftime("%Y-%m-%d %H:%M")
        for i in edited_rows:
            ind = index_[i]
            base.loc[ind, edited_rows[i].keys()] = edited_rows[i].values()
    else:
        base = pd.DataFrame(True, index=dat.index, columns=collabs.columns)
    return base


# Fillers


def old_vol_col(value, df_collabs, personne):
    res = 0
    if value == "📞 Appel":
        res = df_collabs.loc["appel/heure", personne]  # ajouter la personne
    elif value == "📧 Mail":
        res = df_collabs.loc["mail/heure", personne]
    return res


def vol_col(value, df_collabs, personne, etat):
    res = 0
    if value[2:] == etat:
        res = df_collabs.loc[etat + "/heure", personne]  # ajouter la personne
    return res


def col_theor(df, df_collabs, etat):
    b = pd.DataFrame()
    for column in df_collabs.columns:
        b = pd.concat(
            [b, df[column].apply(vol_col, args=(df_collabs, column, etat))], axis=1
        )
    b = b.apply(sum, axis=1)
    return b


def filler(dat, df_collabs, model, m_a, m, m_m, n_col):
    pred = model.predict(dat).astype(int)
    pred_ = pd.DataFrame(pred, index=dat.index)
    planning = pd.DataFrame("Appel", index=dat.index, columns=df_collabs.columns)
    # cf rhodia notebook to add columns : pred mail, status of the resolution of the problem for this hour
    for i in planning.index:
        a = pred_.loc[i]  # pred à i
        n_app, n_mai, statut, obj = solve_planning_problem(a, m_a, m, m_m, n_col)
        planning.loc[i, planning.columns[-int(n_mai):]] = "Mail"
        planning.loc[i, "Statut"] = statut
    planning[
        "Prédiction"
    ] = pred  # Adding pred to the left of the planning to make it appears.
    # Reorganising planning in this order
    cols = list(planning.columns)
    cols.remove("Prédiction")
    cols.remove("Statut")
    new_cols = ["Prédiction", "Statut"] + cols
    planning = planning.reindex(columns=new_cols)
    return planning


def filler_team(dat, df_presence, df_collabs, model, m_a, m, m_m, e_col, p_col):
    pred = model.predict(dat).astype(int)
    pred_ = pd.DataFrame(pred, index=dat.index)  # mettre les pred par équipe
    # idem pour mail
    planning = pd.DataFrame(
        "📞 Appel", index=dat.index, columns=df_presence.columns
    )  # remplacer par df_presence avec la bonne sélection mais commencer par la nouvelle version de
    # recommandation par utilisateur car devrait tout changer
    # et problablement prendre en compte directement
    for i in planning.index:  # le remplacer par un apply
        a = int(pred_.loc[i] * p_col)  # pred à i
        n_app, n_mai, statut, obj = solve_planning_problem(a, m_a, m, m_m, e_col)
        planning.loc[i, planning.columns[-int(n_mai):]] = "📧 Mail"
        planning.loc[i, "Statut"] = statut
    planning["#Appel"] = (
        pred * p_col
    )  # Adding pred to the left of the planning to make it appears.
    planning["#Appel"] = planning["#Appel"].astype(type(a))
    planning[
        "Mail à traiter"
    ] = m  # Adding numbers of mail to the left of the planning to make it appears.
    # Reorganising planning in this order
    cols = list(planning.columns)
    cols.remove("#Appel")
    cols.remove("Statut")
    cols.remove("Mail à traiter")
    new_cols = ["#Appel", "Mail à traiter", "Statut"] + cols
    planning = planning.reindex(columns=new_cols)

    # Change le type des colonnes en catégories pour permettre une selectbox dans streamlit
    cat_type = CategoricalDtype(categories=["📞 Appel", "📧 Mail"])  # , ordered=True)
    planning.loc[:, df_presence.columns] = planning.loc[:, df_presence.columns].astype(
        cat_type
    )

    # Ajoute les colonnes de calculs théoriques des volumes réalisés après sizing
    planning["§Appel"] = col_theor(planning, df_collabs, "Appel")
    planning["§Mail"] = col_theor(planning, df_collabs, "Mail")

    # Ajoute la QS
    planning["QS Appel"] = (planning["§Appel"] * 100 / planning["#Appel"]).astype("int")
    planning["QS Mail"] = (planning["§Mail"] * 100 / planning["Mail à traiter"]).astype(
        "int"
    )
    return planning


# Planning styler


def color_background(value):
    text_color = "black"
    background_color = "white"
    if value == "📧 Mail":
        background_color = "lightblue"
    elif value == "Infeasible":
        text_color = "white"
        background_color = "grey"
    return f"color: {text_color}; background-color: {background_color}"


# ajouter les fonctions de style : 1 pour remplacer infeasible,
# une autre pour les mettre les emojis dans les cellules si ce n'est pas déjà fait avant
