# ============================================================
# APPLICATION STREAMLIT
# MODELE DIFFERENTIEL DJE-GUEU
# Outil exploratoire d'aide à la décision
# Région de Labé - République de Guinée
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# 1. CONFIGURATION DE LA PAGE
# ============================================================

st.set_page_config(


    page_title="DJE-Gueu | Région de Labé",
    page_icon="🖥️",
    layout="wide"
)


# ============================================================
# 2. TITRE DE L'APPLICATION
# ============================================================
#Afficher le logo dans la barre latérale (Sidebar)
st.sidebar.image("logo_DLR.PNG", use_container_width=True)
st.sidebar.title("Navigation")

#Afficher le logo dans la page principale
st.image("logo_DLR.PNG", width=200)
st.title("    Modèle")
st.title("🗺️ DJE-Gueu")
st.subheader(
    "Outil numérique exploratoire pour l'analyse de "
    "l'insertion socioéconomique des jeunes de Labé"
)

st.markdown(
    """
    🔎 **Territoire étudié : Région de Labé, République de Guinée**

    Cette application permet d'explorer mathématiquement l'effet des
    politiques d'activation économique sur la dynamique de la population
    jeune économiquement active.

    🔎 **Information**

    Les travaux de recherche scientifique ont été réalisés par Guy Ghislain GUEU, 
    Scientifique indépendant en équations différentielles.

    🔎 **Contacts**
    
    (00 221) 77 807 62 07 uniquement par WhatsApp
    ghislainci@outlook.fr
    """
)


# ============================================================
# 3. MODELE DJE-GUEU
# ============================================================

def modele_dje_gueu(Y, lam, alpha, kappa, mu):
    """
    Modèle différentiel DJE-Gueu :

    dY/dt =
        lambda * Y
        + alpha * Y * (1 - exp(-kappa * Y))
        - mu * Y**2
    """

    return (
        lam * Y
        + alpha * Y * (1 - np.exp(-kappa * Y))
        - mu * Y**2
    )


# ============================================================
# 4. METHODE D'EULER EXPLICITE
# ============================================================

def euler_dje_gueu(
    Y0,
    lam,
    alpha,
    kappa,
    mu,
    h,
    T
):
    """
    Résolution numérique du modèle DJE-Gueu
    par la méthode d'Euler explicite.
    """

    N = int(T / h)

    t = np.linspace(0, T, N + 1)

    Y = np.zeros(N + 1)

    Y[0] = Y0

    for n in range(N):

        f = modele_dje_gueu(
            Y[n],
            lam,
            alpha,
            kappa,
            mu
        )

        Y[n + 1] = Y[n] + h * f

        # Eviter des valeurs négatives
        # pour une interprétation démographique
        if Y[n + 1] < 0:
            Y[n + 1] = 0

    return t, Y


# ============================================================
# 5. CALCUL DE LA STABILITE
# ============================================================

def derivee_dynamique(Y, lam, alpha, kappa, mu):

    return (
        lam
        + alpha * (
            1
            - np.exp(-kappa * Y)
            + kappa * Y * np.exp(-kappa * Y)
        )
        - 2 * mu * Y
    )


def trouver_equilibre(
    lam,
    alpha,
    kappa,
    mu,
    Y_max=10000
):

    Y_values = np.linspace(
        0,
        Y_max,
        10000
    )

    F = (
        lam
        + alpha * (
            1 - np.exp(-kappa * Y_values)
        )
        - mu * Y_values
    )

    changements = []

    for i in range(len(Y_values) - 1):

        if F[i] * F[i + 1] < 0:

            Y_eq = (
                Y_values[i]
                + Y_values[i + 1]
            ) / 2

            changements.append(Y_eq)

    return changements


# ============================================================
# 6. SIDEBAR - PARAMETRES
# ============================================================

st.sidebar.header("⚙️ Paramètres du modèle")


Y0 = st.sidebar.number_input(
    "Population initiale Y₀",
    min_value=0.0,
    value=100.0,
    step=10.0
)


lam = st.sidebar.number_input(
    "λ — Renouvellement",
    min_value=0.0,
    value=0.02,
    step=0.01,
    format="%.4f"
)


alpha = st.sidebar.slider(
    "α — Intensité des politiques d'activation",
    min_value=0.0,
    max_value=1.0,
    value=0.10,
    step=0.01
)


kappa = st.sidebar.number_input(
    "κ — Vitesse de diffusion",
    min_value=0.0001,
    value=0.01,
    step=0.005,
    format="%.4f"
)


mu = st.sidebar.number_input(
    "μ — Saturation économique",
    min_value=0.0001,
    value=0.001,
    step=0.0005,
    format="%.5f"
)


T = st.sidebar.slider(
    "Horizon de simulation",
    min_value=10,
    max_value=500,
    value=100,
    step=10
)


h = st.sidebar.number_input(
    "Pas numérique h",
    min_value=0.001,
    max_value=1.0,
    value=0.01,
    step=0.01
)


# ============================================================
# 7. SIMULATION
# ============================================================

t, Y = euler_dje_gueu(
    Y0,
    lam,
    alpha,
    kappa,
    mu,
    h,
    T
)


# ============================================================
# 8. INDICATEURS PRINCIPAUX
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Y initial",
        f"{Y[0]:.2f}"
    )


with col2:
    st.metric(
        "Y final",
        f"{Y[-1]:.2f}"
    )


with col3:

    variation = Y[-1] - Y[0]

    st.metric(
        "Variation",
        f"{variation:.2f}"
    )


with col4:

    vitesse_finale = modele_dje_gueu(
        Y[-1],
        lam,
        alpha,
        kappa,
        mu
    )

    st.metric(
        "dY/dt final",
        f"{vitesse_finale:.4f}"
    )


# ============================================================
# 9. GRAPHIQUE PRINCIPAL
# ============================================================

st.header(
    "📈 Dynamique de la population jeune économiquement active"
)


fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.plot(
    t,
    Y,
    label="Y(t)"
)

ax.set_xlabel(
    "Temps"
)

ax.set_ylabel(
    "Population jeune économiquement active"
)

ax.set_title(
    "Simulation du modèle DJE-Gueu"
)

ax.grid(True)

ax.legend()

st.pyplot(fig)


# ============================================================
# 10. EQUILIBRES
# ============================================================

st.header(
    "⚖️ Analyse des équilibres"
)

equilibres = trouver_equilibre(
    lam,
    alpha,
    kappa,
    mu
)

st.write(
    "L'équilibre trivial du modèle est : Y* = 0."
)

if len(equilibres) > 0:

    for i, eq in enumerate(
        equilibres,
        start=1
    ):

        stabilite = derivee_dynamique(
            eq,
            lam,
            alpha,
            kappa,
            mu
        )

        if stabilite < 0:
            statut = "Stable localement"
        else:
            statut = "Instable localement"

        st.write(
            f"Équilibre positif {i} : "
            f"Y* ≈ {eq:.2f} — {statut}"
        )

else:

    st.info(
        "Aucun équilibre positif détecté "
        "dans l'intervalle étudié."
    )


# ============================================================
# 11. COMPARAISON DES SCENARIOS
# ============================================================

st.header(
    "🔎 Comparaison des scénarios de politique publique"
)

st.write(
    """
    Trois scénarios sont proposés :

    - Faible activation économique
    - Activation intermédiaire
    - Forte activation économique
    """
)


alpha_faible = st.number_input(
    "α — Scénario faible",
    min_value=0.0,
    value=0.05,
    step=0.01
)


alpha_intermediaire = st.number_input(
    "α — Scénario intermédiaire",
    min_value=0.0,
    value=0.10,
    step=0.01
)


alpha_forte = st.number_input(
    "α — Scénario fort",
    min_value=0.0,
    value=0.20,
    step=0.01
)


scenarios = {

    "Faible activation":
        alpha_faible,

    "Activation intermédiaire":
        alpha_intermediaire,

    "Forte activation":
        alpha_forte
}


fig2, ax2 = plt.subplots(
    figsize=(10, 5)
)


resultats = []


for nom, alpha_scenario in scenarios.items():

    t_s, Y_s = euler_dje_gueu(
        Y0,
        lam,
        alpha_scenario,
        kappa,
        mu,
        h,
        T
    )

    ax2.plot(
        t_s,
        Y_s,
        label=nom
    )

    resultats.append({

        "Scénario": nom,

        "Alpha":
            alpha_scenario,

        "Y initial":
            Y_s[0],

        "Y final":
            Y_s[-1],

        "Variation":
            Y_s[-1] - Y_s[0]

    })


ax2.set_xlabel(
    "Temps"
)

ax2.set_ylabel(
    "Population jeune économiquement active"
)

ax2.set_title(
    "Comparaison des scénarios d'activation"
)

ax2.grid(True)

ax2.legend()

st.pyplot(fig2)


# ============================================================
# 12. TABLEAU COMPARATIF
# ============================================================

st.subheader(
    "Tableau comparatif"
)

df_resultats = pd.DataFrame(
    resultats
)

st.dataframe(
    df_resultats,
    use_container_width=True
)


# ============================================================
# 13. ANALYSE DE SENSIBILITE
# ============================================================

st.header(
    "📊 Analyse de sensibilité"
)


st.write(
    """
    Cette section permet d'étudier l'influence du paramètre α
    sur la valeur finale de Y(T), toutes choses égales par ailleurs.
    """
)


alpha_min = st.slider(
    "α minimum",
    0.0,
    1.0,
    0.0,
    0.01
)


alpha_max = st.slider(
    "α maximum",
    0.01,
    2.0,
    0.5,
    0.01
)


nombre_scenarios = st.slider(
    "Nombre de scénarios",
    5,
    50,
    20
)


alphas = np.linspace(
    alpha_min,
    alpha_max,
    nombre_scenarios
)


Y_finaux = []


for a in alphas:

    _, Y_sens = euler_dje_gueu(
        Y0,
        lam,
        a,
        kappa,
        mu,
        h,
        T
    )

    Y_finaux.append(
        Y_sens[-1]
    )


fig3, ax3 = plt.subplots(
    figsize=(10, 5)
)


ax3.plot(
    alphas,
    Y_finaux,
    marker="o"
)


ax3.set_xlabel(
    "α — Intensité de l'activation économique"
)

ax3.set_ylabel(
    "Y(T)"
)

ax3.set_title(
    "Analyse de sensibilité de Y(T) au paramètre α"
)

ax3.grid(True)

st.pyplot(fig3)


# ============================================================
# 14. INTERPRETATION
# ============================================================

st.header(
    "🧭 Interprétation pour l'aide à la décision"
)


st.markdown(
    f"""
    ### Résultat de la simulation

    Pour les paramètres actuellement sélectionnés :

    - **Y₀** = {Y0:.2f}
    - **λ** = {lam:.4f}
    - **α** = {alpha:.4f}
    - **κ** = {kappa:.4f}
    - **μ** = {mu:.5f}

    La simulation donne une valeur finale :

    **Y(T) = {Y[-1]:.2f}**

    ### Lecture politique

    Le modèle DJE-Gueu suggère que l'augmentation de **α**,
    représentant l'intensité des politiques d'activation économique,
    tend à renforcer la dynamique d'insertion lorsque Y > 0.

    Cependant, cette dynamique est limitée par le mécanisme de
    saturation représenté par **μY²**.

    En conséquence, une politique d'insertion efficace ne devrait pas
    seulement augmenter l'activation des jeunes. Elle devrait également
    renforcer la capacité d'absorption économique du territoire.

    Les leviers peuvent notamment concerner :

    - la formation professionnelle ;
    - l'entrepreneuriat ;
    - l'accès au financement ;
    - l'agriculture ;
    - l'élevage ;
    - l'artisanat ;
    - le commerce ;
    - les services ;
    - les chaînes de valeur ;
    - les infrastructures économiques.
    """
)


# ============================================================
# 15. AVERTISSEMENT FINAL
# ============================================================

st.warning(
    """
    IMPORTANT :

    Le modèle DJE-Gueu constitue ici un outil mathématique exploratoire
    d'aide à l'analyse.

    

    Pour une utilisation institutionnelle, les paramètres λ, α, κ et μ
    devront être estimés ou calibrés à partir de données socioéconomiques
    réelles, documentées et régulièrement actualisées.
    """
)


# ============================================================
# FIN DE L'APPLICATION
# ============================================================
