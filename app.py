# -*- coding: utf-8 -*-
"""
Planejamento Estratégico Empresarial - Seleção de Portfólio de Projetos com MCDA (PROMETHEE II + V)
App Streamlit para apoiar a decisão de portfólio de investimentos alinhada
ao planejamento estratégico empresarial.

Versão 3 — os SEIS tipos de funções de preferência generalizadas de
Brans, Vincke e Mareschal (1986), com a nomenclatura:

  Tipo 1 - Critério Usual                                        (Usual criterion)          sem parâmetros
  Tipo 2 - Quasi-Critério (U-Shape)                              (Quasi criterion)          parâmetro q
  Tipo 3 - Critério Linear (V-Shape)                             (Linear criterion)         parâmetro p
  Tipo 4 - Critério de Nível                                     (Level criterion)          parâmetros q, p
  Tipo 5 - Critério Linear com Indiferença (V-Shape w/ Indif.)   (Linear with indifference) parâmetros q, p
  Tipo 6 - Critério Gaussiano                                    (Gaussian criterion)       parâmetro s

Como funciona o app:
1) PROMETHEE II ordena os projetos (fluxo líquido de preferência, phi)
2) PROMETHEE V seleciona a carteira que maximiza o phi total sob restrição
   de orçamento (problema da mochila 0-1), respeitando projetos obrigatórios.
"""

import io
import math
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Configuração da página
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Planejamento Estratégico Empresarial — Seleção de Portfólio de Projetos de Investimento com MCDA PROMETHEE V",
    page_icon="📊",
    layout="wide",
)

# Rótulos oficiais (aparecem no dropdown, na planilha e nos resultados)
T1 = "Tipo 1 - Critério Usual"
T2 = "Tipo 2 - Quasi-Critério (U-Shape)"
T3 = "Tipo 3 - Critério Linear (V-Shape)"
T4 = "Tipo 4 - Critério de Nível"
T5 = "Tipo 5 - Critério Linear com Indiferença (V-Shape with Indifference)"
T6 = "Tipo 6 - Critério Gaussiano"

FUNCOES_VALIDAS = [T1, T2, T3, T4, T5, T6]

DESCRICAO_FUNCOES = {
    T1: "Usual criterion — qualquer diferença positiva gera preferência total (P = 1). Sem parâmetros. Indicado para escalas discretas (ex.: notas 1–5) em que qualquer diferença conta.",
    T2: "Quasi criterion — diferenças até q são indiferentes (P = 0); acima de q, preferência total (P = 1). Parâmetro: q_indiferenca.",
    T3: "Linear criterion — a preferência cresce linearmente de 0 até p (P = d/p) e é total acima de p. Parâmetro: p_preferencia. Sem faixa de indiferença.",
    T4: "Level criterion — P = 0 até q; P = 1/2 entre q e p; P = 1 acima de p. Parâmetros: q_indiferenca e p_preferencia (p > q). Útil para julgamentos em degraus (indiferente / preferência fraca / preferência forte).",
    T5: "Linear with indifference — P = 0 até q; cresce linearmente (P = (d−q)/(p−q)) entre q e p; P = 1 acima de p. Parâmetros: q_indiferenca e p_preferencia. O mais usado para critérios contínuos (ex.: VPL).",
    T6: "Gaussian criterion — P = 1 − exp(−d²/2s²) para d > 0: crescimento suave, sem descontinuidades. Parâmetro: s_gaussiana (ponto de inflexão).",
}

# ----------------------------------------------------------------------------
# Dados de exemplo (usados se o usuário não subir planilha)
# ----------------------------------------------------------------------------
def dados_exemplo():
    criterios = pd.DataFrame({
        "Criterio": ["VPL (R$ mil)", "Alinhamento Estratégico (1-5)",
                     "Risco de Execução (1-5)", "Impacto ESG (1-5)"],
        "Peso": [0.35, 0.30, 0.20, 0.15],
        "Objetivo": ["max", "max", "min", "max"],
        "FuncaoPreferencia": [T5, T3, T1, T3],
        "q_indiferenca": [50.0, 0.0, 0.0, 0.0],
        "p_preferencia": [500.0, 2.0, 0.0, 2.0],
        "s_gaussiana": [0.0, 0.0, 0.0, 0.0],
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
# Núcleo PROMETHEE — as seis funções de preferência generalizadas
# (Brans, Vincke & Mareschal, 1986)
# ----------------------------------------------------------------------------
def normalizar_tipo(tipo):
    """
    Converte o rótulo da função (nomenclatura oficial, sinônimos em PT/EN
    ou os nomes curtos das versões anteriores do app) na chave interna t1..t6.
    """
    t = str(tipo).strip().lower()
    t = (t.replace("_", "-").replace("í", "i").replace("é", "e")
           .replace("ç", "c").replace("ã", "a"))
    compacto = t.replace(" ", "").replace("-", "")

    # 1) Prefixo "Tipo N" tem prioridade máxima (nomenclatura oficial)
    for n, chave in [("1", "t1"), ("2", "t2"), ("3", "t3"),
                     ("4", "t4"), ("5", "t5"), ("6", "t6")]:
        if compacto.startswith(f"tipo{n}") or compacto == n:
            return chave

    # 2) Palavras-chave (sinônimos PT/EN e nomes curtos das versões anteriores)
    if "indif" in compacto:                      # linear com indiferença
        return "t5"
    if "gauss" in compacto:
        return "t6"
    if "quasi" in compacto or "quase" in compacto or "ushape" in compacto:
        return "t2"
    if "nivel" in compacto or "level" in compacto or "patamar" in compacto:
        return "t4"
    if "vshape" in compacto:                     # v-shape sem 'indif' = Tipo 3
        return "t3"
    if "linear" in compacto:
        # Compatibilidade: nas versões anteriores, "linear" era o Tipo 5
        # (com q e p). Como o Tipo 5 com q=0 é matematicamente idêntico ao
        # Tipo 3, mapear para t5 preserva os dois comportamentos.
        return "t5"
    if "usual" in compacto:
        return "t1"
    return "t1"  # fallback defensivo


def funcao_preferencia(d, tipo, q=0.0, p=0.0, s=0.0):
    """
    Grau de preferência P(d) para uma diferença de desempenho d.
    Implementa os 6 critérios generalizados de Brans, Vincke & Mareschal (1986).
    Parâmetros ausentes/zerados degradam graciosamente para o tipo mais simples.
    """
    if d <= 0:
        return 0.0
    chave = normalizar_tipo(tipo)
    q = max(float(q or 0.0), 0.0)
    p = max(float(p or 0.0), 0.0)
    s = max(float(s or 0.0), 0.0)

    if chave == "t1":                          # Tipo 1 — Critério Usual
        return 1.0

    if chave == "t2":                          # Tipo 2 — Quasi-Critério (U-Shape)
        return 0.0 if d <= q else 1.0

    if chave == "t3":                          # Tipo 3 — Critério Linear (V-Shape)
        if p <= 0:
            return 1.0                         # sem p, degrada para usual
        return min(d / p, 1.0)

    if chave == "t4":                          # Tipo 4 — Critério de Nível
        if p <= q:                             # parâmetros inconsistentes
            return (0.0 if d <= q else 1.0) if q > 0 else 1.0
        if d <= q:
            return 0.0
        if d <= p:
            return 0.5
        return 1.0

    if chave == "t5":                          # Tipo 5 — Linear com Indiferença
        if p <= 0:
            return 1.0                         # sem p, degrada para usual
        if d <= q:
            return 0.0
        if d >= p:
            return 1.0
        if p > q:
            return (d - q) / (p - q)
        return 1.0

    if chave == "t6":                          # Tipo 6 — Critério Gaussiano
        if s <= 0:
            return 1.0                         # sem s, degrada para usual
        return 1.0 - math.exp(-(d ** 2) / (2.0 * s ** 2))

    return 1.0  # fallback defensivo


def promethee_ii(projetos, criterios, col_nome="Projeto"):
    """Calcula fluxos phi+, phi- e phi líquido (PROMETHEE II)."""
    nomes = projetos[col_nome].tolist()
    n = len(nomes)
    soma_pesos = criterios["Peso"].astype(float).sum()

    pi = np.zeros((n, n))  # índice de preferência agregado pi(a,b)
    for _, cr in criterios.reset_index(drop=True).iterrows():
        col = cr["Criterio"]
        vals = projetos[col].astype(float).values
        sinal = 1.0 if str(cr["Objetivo"]).strip().lower().startswith("max") else -1.0
        q = float(cr.get("q_indiferenca", 0) or 0)
        p = float(cr.get("p_preferencia", 0) or 0)
        s = float(cr.get("s_gaussiana", 0) or 0)
        w = float(cr["Peso"]) / soma_pesos
        tipo = cr["FuncaoPreferencia"]
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                d = sinal * (vals[a] - vals[b])
                pi[a, b] += w * funcao_preferencia(d, tipo, q, p, s)

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
                cst = custos[list(comb)].sum()
                if cst <= saldo:
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
            "   - FuncaoPreferencia: um dos 6 tipos de Brans, Vincke & Mareschal (1986):",
            f"       * {T1} (Usual criterion) — qualquer diferença gera preferência total. Sem parâmetros.",
            f"       * {T2} (Quasi criterion) — indiferente até q; preferência total acima de q. Usa: q_indiferenca.",
            f"       * {T3} (Linear criterion) — preferência cresce linearmente de 0 até p (P = d/p). Usa: p_preferencia.",
            f"       * {T4} (Level criterion) — 0 até q; 1/2 entre q e p; 1 acima de p. Usa: q_indiferenca e p_preferencia (p > q).",
            f"       * {T5} (Linear with indifference) — 0 até q; cresce linearmente entre q e p; 1 acima de p. Usa: q e p.",
            f"       * {T6} (Gaussian criterion) — crescimento suave P = 1 - exp(-d²/2s²). Usa: s_gaussiana.",
            "   - q_indiferenca: limiar de indiferença (Tipos 2, 4 e 5; deixe 0 nos demais).",
            "   - p_preferencia: limiar de preferência total (Tipos 3, 4 e 5; deixe 0 nos demais).",
            "   - s_gaussiana: ponto de inflexão s (apenas Tipo 6; deixe 0 nos demais).",
            "   - Pode-se escrever apenas 'Tipo 1' ... 'Tipo 6' — o app reconhece. Parâmetro exigido deixado em 0",
            "     faz a função degradar para o Critério Usual (o app avisa).",
            "2) Aba 'Projetos': uma linha por projeto candidato.",
            "   - Custo: investimento requerido (mesma unidade do orçamento informado no app).",
            "   - Obrigatorio: 'sim' para projetos mandatórios (regulatórios, segurança) que entram antes da otimização.",
            "   - Demais colunas: desempenho do projeto em CADA critério — os nomes devem ser IDÊNTICOS aos da aba Criterios.",
            "3) Salve e faça upload no app. Informe o orçamento e navegue pelas abas de resultado.",
        ]}).to_excel(wr, sheet_name="Instrucoes", index=False)
    buf.seek(0)
    return buf


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------
st.title("📊 Planejamento Estratégico Empresarial")
st.caption("Seleção de portfólio de projetos alinhada ao planejamento estratégico — "
           "método multicritério **PROMETHEE II + V**, com os 6 tipos de funções de "
           "preferência de Brans, Vincke & Mareschal (1986)")

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
    st.divider()
    with st.expander("📖 As 6 funções de preferência"):
        for nome in FUNCOES_VALIDAS:
            st.markdown(f"**{nome}**  \n{DESCRICAO_FUNCOES[nome]}")

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

# Compatibilidade: garante as colunas de parâmetros (planilhas de versões anteriores)
for col_param in ["q_indiferenca", "p_preferencia", "s_gaussiana"]:
    if col_param not in criterios.columns:
        criterios[col_param] = 0.0
if "Obrigatorio" not in projetos.columns:
    projetos["Obrigatorio"] = "não"

# Normaliza os rótulos das funções para a nomenclatura oficial
# (aceita 'Tipo N', nomes antigos 'usual'/'linear'/'v-shape' etc. e sinônimos EN)
ROTULO_POR_CHAVE = {"t1": T1, "t2": T2, "t3": T3, "t4": T4, "t5": T5, "t6": T6}
criterios["FuncaoPreferencia"] = criterios["FuncaoPreferencia"].apply(
    lambda v: ROTULO_POR_CHAVE[normalizar_tipo(v)]
)

# Validações básicas
faltando = [x for x in criterios["Criterio"] if x not in projetos.columns]
if faltando:
    st.error(f"Colunas de critério ausentes na aba Projetos: {faltando}. "
             "Os nomes devem ser idênticos aos da aba Criterios.")
    st.stop()

# Avisos de parametrização das funções de preferência
avisos = []
for _, cr in criterios.iterrows():
    chave = normalizar_tipo(cr["FuncaoPreferencia"])
    q = float(cr.get("q_indiferenca", 0) or 0)
    p = float(cr.get("p_preferencia", 0) or 0)
    s = float(cr.get("s_gaussiana", 0) or 0)
    nome = cr["Criterio"]
    if chave == "t2" and q <= 0:
        avisos.append(f"'{nome}': Tipo 2 (Quasi-Critério) sem q_indiferenca > 0 — comporta-se como o Tipo 1 (Usual).")
    if chave in ("t3", "t5") and p <= 0:
        avisos.append(f"'{nome}': {ROTULO_POR_CHAVE[chave]} sem p_preferencia > 0 — tratado como Tipo 1 (Usual).")
    if chave == "t4" and p <= q:
        avisos.append(f"'{nome}': Tipo 4 (Nível) exige p_preferencia > q_indiferenca — tratado como degrau simples.")
    if chave == "t5" and 0 < p <= q:
        avisos.append(f"'{nome}': Tipo 5 com p ≤ q — tratado como Tipo 2 (Quasi-Critério).")
    if chave == "t6" and s <= 0:
        avisos.append(f"'{nome}': Tipo 6 (Gaussiano) sem s_gaussiana > 0 — tratado como Tipo 1 (Usual).")
for a in avisos:
    st.warning(a)

tab1, tab2, tab3 = st.tabs(["1️⃣ Dados & Pesos", "2️⃣ Ranking (PROMETHEE II)",
                            "3️⃣ Carteira Ótima (PROMETHEE V)"])

with tab1:
    st.subheader("Critérios de decisão (edite pesos, funções e parâmetros)")
    st.caption("Parâmetros por tipo: q_indiferenca (Tipos 2, 4 e 5) • "
               "p_preferencia (Tipos 3, 4 e 5) • s_gaussiana (Tipo 6). "
               "O Tipo 1 não usa parâmetros.")
    criterios = st.data_editor(
        criterios, num_rows="fixed", use_container_width=True, key="edit_criterios",
        column_config={
            "FuncaoPreferencia": st.column_config.SelectboxColumn(
                "FuncaoPreferencia", options=FUNCOES_VALIDAS, required=True,
                width="large",
                help="Um dos 6 tipos de Brans, Vincke & Mareschal (1986)."
            ),
        },
    )
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
               "pesos dos critérios e pelas funções de preferência. φ > 0: mais forças que fraquezas.")
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
            criterios.to_excel(wr, sheet_name="Criterios_Utilizados", index=False)
        buf.seek(0)
        st.download_button("⬇️ Exportar resultados (.xlsx)", data=buf,
                           file_name="resultado_portfolio_mcda.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.divider()
st.caption("Método: Brans & Vincke (1985); Brans, Vincke & Mareschal (1986) — PROMETHEE II/V, "
           "com os 6 critérios generalizados. Ferramenta de apoio à decisão — os pesos, funções "
           "e julgamentos são responsabilidade dos decisores. "
           "Produto Tecnológico."
           "Contato: Eliezer Guimarães Miranda"
           "E-mail:   eliezer.guimaraes.miranda@gmail.com")
