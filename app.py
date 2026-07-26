# -*- coding: utf-8 -*-
"""
Do Plano à Carteira — Seleção de Portfólio de Projetos com MCDA (PROMETHEE II + V)
App Streamlit para apoiar a decisão de portfólio de investimentos alinhada
ao planejamento estratégico empresarial.

Como funciona:
1) PROMETHEE II ordena os projetos (fluxo líquido de preferência, phi)
2) PROMETHEE V seleciona a carteira que maximiza o phi total sob restrição
   de orçamento (problema da mochila 0-1), respeitando projetos obrigatórios.
"""

import io
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Configuração da página
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Do Plano à Carteira — MCDA PROMETHEE V",
    page_icon="📊",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Dados de exemplo (usados se o usuário não subir planilha)
# ----------------------------------------------------------------------------
def dados_exemplo():
    criterios = pd.DataFrame({
        "Criterio": ["VPL (R$ mil)", "Alinhamento Estratégico (1-5)",
                     "Risco de Execução (1-5)", "Impacto ESG (1-5)"],
        "Peso": [0.35, 0.30, 0.20, 0.15],
        "Objetivo": ["max", "max", "min", "max"],
        "FuncaoPreferencia": ["linear", "linear", "usual", "linear"],
        "q_indiferenca": [50.0, 0.0, 0.0, 0.0],
        "p_preferencia": [500.0, 2.0, 0.0, 2.0],
    })
    projetos = pd.DataFrame({
        "Projeto": ["ERP Cloud", "Nova Linha de Produção", "Expansão Nordeste",
                    "Programa ESG", "Automação Logística", "CRM Comercial",
                    "Eficiência Energética", "Data Analytics"],
        "Custo": [1200, 3500, 2800, 800, 1500, 600, 900, 700],
        "Obrigatorio": ["não", "não", "não", "sim", "não", "não", "não", "não"],
        "VPL (R$ mil)": [900, 2200, 1800, 150, 1100, 450, 600, 500],
        "Alinhamento Estratégico (1-5)": [4, 5, 4, 3, 3, 2, 3, 4],
        "Risco de Execução (1-5)": [3, 4, 4, 1, 2, 1, 2, 2],
        "Impacto ESG (1-5)": [2, 2, 3, 5, 3, 1, 5, 2],
    })
    return criterios, projetos


# ----------------------------------------------------------------------------
# Núcleo PROMETHEE
# ----------------------------------------------------------------------------
def funcao_preferencia(d, tipo, q, p):
    """Grau de preferência P(d) para uma diferença de desempenho d >= 0."""
    if d <= 0:
        return 0.0
    tipo = str(tipo).strip().lower()
    if tipo == "usual":
        return 1.0
    if tipo == "linear":  # V-shape com indiferença (tipo V de Brans)
        if p is None or p <= 0:
            return 1.0
        if d <= q:
            return 0.0
        if d >= p:
            return 1.0
        return (d - q) / (p - q) if p > q else 1.0
    return 1.0  # fallback: usual


def promethee_ii(projetos, criterios, col_nome="Projeto"):
    """Calcula fluxos phi+, phi- e phi líquido (PROMETHEE II)."""
    nomes = projetos[col_nome].tolist()
    n = len(nomes)
    pesos = criterios["Peso"].astype(float).values
    pesos = pesos / pesos.sum()  # normaliza

    pi = np.zeros((n, n))  # índice de preferência agregado pi(a,b)
    for _, c in criterios.reset_index(drop=True).iterrows():
        col = c["Criterio"]
        vals = projetos[col].astype(float).values
        sinal = 1.0 if str(c["Objetivo"]).strip().lower().startswith("max") else -1.0
        q = float(c.get("q_indiferenca", 0) or 0)
        p = float(c.get("p_preferencia", 0) or 0)
        w = float(c["Peso"]) / criterios["Peso"].astype(float).sum()
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                d = sinal * (vals[a] - vals[b])
                pi[a, b] += w * funcao_preferencia(d, c["FuncaoPreferencia"], q, p)

    phi_pos = pi.sum(axis=1) / (n - 1)
    phi_neg = pi.sum(axis=0) / (n - 1)
    phi = phi_pos - phi_neg
    return pd.DataFrame({
        col_nome: nomes,
        "phi+ (força)": np.round(phi_pos, 4),
        "phi- (fraqueza)": np.round(phi_neg, 4),
        "phi líquido": np.round(phi, 4),
    }).sort_values("phi líquido", ascending=False).reset_index(drop=True)


def promethee_v(ranking, projetos, orcamento, col_nome="Projeto"):
    """
    PROMETHEE V: escolhe a carteira que maximiza a soma dos fluxos líquidos
    (phi) sob restrição de orçamento (mochila 0-1), com projetos obrigatórios.
    Busca exata: força bruta até 20 projetos livres; acima disso, programação
    dinâmica com custos discretizados.
    """
    base = ranking.merge(
        projetos[[col_nome, "Custo", "Obrigatorio"]], on=col_nome
    )
    base["Obrigatorio"] = (
        base["Obrigatorio"].astype(str).str.strip().str.lower().isin(
            ["sim", "s", "yes", "1", "true", "x"])
    )

    obrig = base[base["Obrigatorio"]]
    livres = base[~base["Obrigatorio"]].reset_index(drop=True)

    custo_obrig = obrig["Custo"].sum()
    if custo_obrig > orcamento:
        return None, custo_obrig  # orçamento insuficiente p/ obrigatórios

    saldo = orcamento - custo_obrig
    custos = livres["Custo"].astype(float).values
    phis = livres["phi líquido"].astype(float).values
    m = len(livres)

    melhor_sel, melhor_val = [], -np.inf

    if m == 0:
        melhor_sel, melhor_val = [], 0.0
    elif m <= 20:
        # força bruta exata
        idx = list(range(m))
        melhor_val = 0.0
        for r in range(1, m + 1):
            for comb in combinations(idx, r):
                c = custos[list(comb)].sum()
                if c <= saldo:
                    v = phis[list(comb)].sum()
                    if v > melhor_val:
                        melhor_val, melhor_sel = v, list(comb)
    else:
        # DP mochila com custos discretizados
        passo = max(1.0, saldo / 5000.0)
        cap = int(saldo / passo)
        c_int = np.maximum(1, np.ceil(custos / passo).astype(int))
        dp = np.full(cap + 1, -np.inf)
        dp[0] = 0.0
        escolha = [[False] * (cap + 1) for _ in range(m)]
        for i in range(m):
            if phis[i] <= 0:
                continue
            for w in range(cap, c_int[i] - 1, -1):
                cand = dp[w - c_int[i]] + phis[i]
                if cand > dp[w]:
                    dp[w] = cand
                    escolha[i][w] = True
        w = int(np.argmax(dp))
        melhor_val = max(dp[w], 0.0)
        for i in range(m - 1, -1, -1):
            if w >= 0 and escolha[i][w]:
                melhor_sel.append(i)
                w -= c_int[i]

    selecionados = pd.concat(
        [obrig, livres.iloc[sorted(melhor_sel)]], ignore_index=True
    ) if len(melhor_sel) or len(obrig) else obrig

    return selecionados, custo_obrig


# ----------------------------------------------------------------------------
# Geração da planilha modelo (download dentro do app)
# ----------------------------------------------------------------------------
def gerar_modelo_xlsx():
    criterios, projetos = dados_exemplo()
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as wr:
        criterios.to_excel(wr, sheet_name="Criterios", index=False)
        projetos.to_excel(wr, sheet_name="Projetos", index=False)
        pd.DataFrame({"Instruções": [
            "1) Aba 'Criterios': liste os critérios de decisão (derivados do seu planejamento estratégico / BSC).",
            "   - Peso: importância relativa (será normalizado automaticamente).",
            "   - Objetivo: 'max' (quanto maior, melhor) ou 'min' (quanto menor, melhor).",
            "   - FuncaoPreferencia: 'usual' (qualquer diferença já é preferência total) ou 'linear' (preferência cresce entre q e p).",
            "   - q_indiferenca: diferença até a qual os projetos são indiferentes (0 se não quiser usar).",
            "   - p_preferencia: diferença a partir da qual a preferência é total (obrigatório se função = linear).",
            "2) Aba 'Projetos': uma linha por projeto candidato.",
            "   - Custo: investimento requerido (mesma unidade do orçamento informado no app).",
            "   - Obrigatorio: 'sim' para projetos mandatórios (regulatórios, segurança) que entram antes da otimização.",
            "   - Demais colunas: desempenho do projeto em CADA critério — os nomes devem ser IDÊNTICOS aos da aba Criterios.",
            "3) Salve e faça upload no app. Informe o orçamento e clique em Calcular.",
        ]}).to_excel(wr, sheet_name="Instrucoes", index=False)
    buf.seek(0)
    return buf


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------
st.title("📊 Do Plano à Carteira")
st.caption("Seleção de portfólio de projetos alinhada ao planejamento estratégico — "
           "método multicritério **PROMETHEE II + V**")

with st.sidebar:
    st.header("⚙️ Entrada de dados")
    st.download_button(
        "⬇️ Baixar planilha modelo (.xlsx)",
        data=gerar_modelo_xlsx(),
        file_name="modelo_dados_portfolio.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    arquivo = st.file_uploader("Suba sua planilha preenchida", type=["xlsx"])
    usar_exemplo = st.toggle("Usar dados de exemplo", value=arquivo is None)
    st.divider()
    st.header("💰 Restrição")
    orcamento = st.number_input(
        "Orçamento total disponível", min_value=0.0, value=6000.0, step=100.0,
        help="Mesma unidade da coluna 'Custo' da planilha (ex.: R$ mil)."
    )

# Carrega dados
erro = None
if arquivo is not None and not usar_exemplo:
    try:
        criterios = pd.read_excel(arquivo, sheet_name="Criterios")
        projetos = pd.read_excel(arquivo, sheet_name="Projetos")
    except Exception as e:
        erro = f"Não consegui ler a planilha: {e}"
        criterios, projetos = dados_exemplo()
else:
    criterios, projetos = dados_exemplo()

if erro:
    st.error(erro + " — usando dados de exemplo.")

# Limpeza defensiva: remove linhas vazias ou de totalização
projetos = projetos.dropna(subset=["Projeto"])
projetos = projetos[~projetos["Projeto"].astype(str).str.upper().str.startswith("TOTAL")]
criterios = criterios.dropna(subset=["Criterio"]).reset_index(drop=True)
projetos = projetos.reset_index(drop=True)

# Validações básicas
faltando = [c for c in criterios["Criterio"] if c not in projetos.columns]
if faltando:
    st.error(f"Colunas de critério ausentes na aba Projetos: {faltando}. "
             "Os nomes devem ser idênticos aos da aba Criterios.")
    st.stop()
if "Obrigatorio" not in projetos.columns:
    projetos["Obrigatorio"] = "não"

tab1, tab2, tab3 = st.tabs(["1️⃣ Dados & Pesos", "2️⃣ Ranking (PROMETHEE II)",
                            "3️⃣ Carteira Ótima (PROMETHEE V)"])

with tab1:
    st.subheader("Critérios de decisão (edite os pesos, se quiser)")
    criterios = st.data_editor(criterios, num_rows="fixed", use_container_width=True,
                               key="edit_criterios")
    soma = criterios["Peso"].astype(float).sum()
    st.info(f"Soma dos pesos = **{soma:.2f}** (será normalizada para 1,00 no cálculo).")
    st.subheader("Projetos candidatos")
    projetos = st.data_editor(projetos, num_rows="dynamic", use_container_width=True,
                              key="edit_projetos")

# Cálculos
ranking = promethee_ii(projetos, criterios)

with tab2:
    st.subheader("Ranking dos projetos — fluxo líquido de preferência (φ)")
    st.caption("φ = 'placar' de cada projeto nas comparações par a par, ponderado pelos "
               "pesos dos critérios. φ > 0: mais forças que fraquezas.")
    st.dataframe(ranking, use_container_width=True, hide_index=True)
    graf = ranking.set_index("Projeto")[["phi líquido"]]
    st.bar_chart(graf)

with tab3:
    st.subheader("Carteira que maximiza a preferência total sob o orçamento")
    carteira, custo_obrig = promethee_v(ranking, projetos, orcamento)
    if carteira is None:
        st.error(f"Os projetos obrigatórios já custam {custo_obrig:,.0f}, acima do "
                 f"orçamento de {orcamento:,.0f}. Aumente o orçamento ou revise os obrigatórios.")
    else:
        custo_total = carteira["Custo"].sum()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Projetos selecionados", f"{len(carteira)} de {len(projetos)}")
        col2.metric("Custo da carteira", f"{custo_total:,.0f}")
        col3.metric("Orçamento utilizado", f"{(custo_total / orcamento * 100 if orcamento else 0):.1f}%")
        col4.metric("Σ φ da carteira", f"{carteira['phi líquido'].sum():.3f}")

        st.dataframe(
            carteira[["Projeto", "Custo", "Obrigatorio", "phi líquido"]]
            .sort_values("phi líquido", ascending=False),
            use_container_width=True, hide_index=True,
        )

        fora = ranking[~ranking["Projeto"].isin(carteira["Projeto"])]
        if len(fora):
            with st.expander("Projetos fora da carteira (e por quê)"):
                fora_m = fora.merge(projetos[["Projeto", "Custo"]], on="Projeto")
                fora_m["Motivo provável"] = np.where(
                    fora_m["phi líquido"] <= 0,
                    "φ ≤ 0: mais fraquezas que forças",
                    "Não coube no orçamento com maior Σφ",
                )
                st.dataframe(fora_m, use_container_width=True, hide_index=True)

        # Exporta resultado
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as wr:
            ranking.to_excel(wr, sheet_name="Ranking_PROMETHEE_II", index=False)
            carteira.to_excel(wr, sheet_name="Carteira_PROMETHEE_V", index=False)
        buf.seek(0)
        st.download_button("⬇️ Exportar resultados (.xlsx)", data=buf,
                           file_name="resultado_portfolio_mcda.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.divider()
st.caption("Método: Brans & Mareschal (PROMETHEE II/V). Ferramenta de apoio à decisão — "
           "os pesos e julgamentos são responsabilidade dos decisores. "
           "Artigo tecnológico associado: 'Do Plano à Carteira' (FUCAPE Business School).")
