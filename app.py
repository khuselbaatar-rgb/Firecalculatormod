"""
Предел огнестойкости клееной деревянной балки (БКП)
======================================================
Методика: Сивенков А.Б. и др. «Огнестойкость деревянных конструкций»,
Академия ГПС МЧС России, 2023 — Пример расчёта №1 (стр. 73-76)

БАТАЛГААЖСАН (2025): Бүх томьёо номын эх PDF (Word, 73-76 хуудас) болон
УНК ПБОЗ-ын слайд (33-34) хоёртой тоогоор шалгагдсан:
  φfM1=3.29(ном:3.29), σfw1=3.1МПа(ном:3.1), Kfф=1.71(ном:1.71),
  Zcr=11мм(ном:11), Пф=10.6мин(ном:10.6), R10(ном:R10) — бүгд таарсан.

Тооцооны дараалал (номын яг дараалал):
  3.1.1 Сбор нагрузок, определение усилий
  3.1.2 Определение геометрических характеристик
  3.1.3 Условие I  — прочность по нормальным напряжениям (изгиб)
  3.1.4 Условие II — прочность по касательным напряжениям (скалывание)
  3.1.5 Условие III — устойчивость плоской формы деформирования
  Итог  — Zcr,min, Пф, класс огнестойкости
"""

import math
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import brentq

st.set_page_config(page_title="Огнестойкость БКП", page_icon="🔥", layout="wide")
st.markdown(
    "<style>[data-testid='stSidebar']{min-width:360px;max-width:400px}</style>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
#  ТАБЛИЦЫ (Табл. 1.1, 1.2, Приложение Д)
# ══════════════════════════════════════════════════════════════════════════════

RFW    = {"1": 29.0, "2": 26.0, "3": 18.0}             # Rfw, МПа (Табл.1.1)
RFQS_G = {"1": 1.3,  "2": 1.2,  "3": 1.1}              # Rfqs клееная (Табл.1.1)
RFQS_S = {"1": 3.7,  "2": 3.2,  "3": 2.9}              # Rfqs цельная (Табл.1.1)
CHAR   = {("Клееная", True): 0.6, ("Клееная", False): 0.7,   # υ, мм/мин (Табл.1.2)
          ("Цельная", True): 0.8, ("Цельная", False): 1.0}
KN     = {"4 стороны": (2, 2), "3 стороны": (2, 1)}    # (k, n) Приложение Д

FIRE_CLS = [(15, "R15"), (30, "R30"), (45, "R45"), (60, "R60"),
            (90, "R90"), (120, "R120"), (180, "R180")]

COND = {
    "I":   "Нормальные напряжения (изгиб)",
    "II":  "Касательные напряжения (скалывание)",
    "III": "Устойчивость плоской формы",
}


def fire_class(t: float) -> str:
    for lim, cls in FIRE_CLS:
        if t <= lim:
            return cls
    return "R180+"


def pf_formula(Z_mm: float, tau0: float, delta_mm: float, v: float) -> float:
    """Формула (1.4)/(1.5): фактический предел огнестойкости, мин."""
    if Z_mm > delta_mm:
        return tau0 + (Z_mm - delta_mm) / v
    return Z_mm * tau0 / delta_mm


def white_layout(height=400, xtitle="", ytitle="", title=""):
    d = dict(
        paper_bgcolor="white", plot_bgcolor="white", height=height,
        margin=dict(l=55, r=20, t=40 if title else 20, b=45),
        xaxis=dict(title=xtitle, color="#333", linecolor="#ccc",
                   gridcolor="#f0f0f0", showgrid=True, zeroline=False),
        yaxis=dict(title=ytitle, color="#333", linecolor="#ccc",
                   gridcolor="#f0f0f0", showgrid=True, zeroline=False),
        legend=dict(font=dict(color="#333", size=12), bgcolor="white",
                    bordercolor="#ddd", borderwidth=1),
        font=dict(family="Arial,sans-serif", color="#333"),
    )
    if title:
        d["title"] = dict(text=title, font=dict(color="#333", size=14))
    return d


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — ИСХОДНЫЕ ДАННЫЕ
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚙️ Исходные данные")

    heating = st.radio("Схема обогрева", ["4 стороны", "3 стороны"], horizontal=True)
    k, n = KN[heating]

    st.divider()
    st.subheader("📐 Геометрия сечения")
    b_mm = st.number_input("b — ширина сечения, мм", 60, 500, 140, 5)
    h_mm = st.number_input("h — высота сечения, мм", 100, 2000, 710, 10)

    st.subheader("📏 Параметры пролёта")
    L_m   = st.number_input("L — расчётный пролёт, м", 1.0, 30.0, 6.0, 0.5)
    a_m   = st.number_input("a — шаг балок, м", 0.5, 10.0, 3.0, 0.25)
    lp_m  = st.number_input("lp — закрепление сжатой кромки, м", 0.5, 20.0, 1.5, 0.25)
    lpf_m = st.number_input("lpf — длина участка без связей при пожаре, м",
                            0.5, 30.0, 3.0, 0.5)

    st.divider()
    st.subheader("📦 Нагрузка")
    q_star  = st.number_input("q* — расчётная нагрузка, кН/м²", 0.5, 50.0, 9.35, 0.05, format="%.2f")
    gamma_f = st.number_input("γf — коэф. надёжности по нагрузке", 1.0, 2.0, 1.2, 0.05, format="%.2f")

    st.divider()
    st.subheader("🪵 Материал")
    sort     = st.selectbox("Сорт древесины", ["1", "2", "3"], index=1)
    material = st.selectbox("Тип материала", ["Клееная", "Цельная"])
    st.caption(
        "💡 Пример №1 книги (стр. 73–76): материал указан как «цельная», "
        "но Rfqs в расчёте Условия II взято для клееной (1.2 МПа) — "
        "несоответствие в тексте оригинала."
    )

    st.divider()
    st.subheader("🔥 Обугливание")
    tau_0  = st.number_input("τ₀ — время до начала обугливания, мин", 1, 10, 4, 1)
    delta  = st.number_input("δ — толщина прогретого слоя, мм", 3, 15, 7, 1)
    tau_os = st.number_input("τос — предел огнестойкости связей, мин",
                              min_value=tau_0 + 1, max_value=120, value=15, step=5)

    st.divider()
    st.caption(
        "Сивенков А.Б. и др.\n"
        "«Огнестойкость деревянных конструкций»\n"
        "Академия ГПС МЧС России, 2023"
    )

# ══════════════════════════════════════════════════════════════════════════════
#  МАТЕРИАЛ И ПРОИЗВОДНЫЕ КОНСТАНТЫ
# ══════════════════════════════════════════════════════════════════════════════

Rfw   = RFW[sort]
Rfqs  = (RFQS_G if material == "Клееная" else RFQS_S)[sort]
b_min = min(b_mm, h_mm)
v     = CHAR[(material, b_min >= 120)]
r     = h_mm / b_mm                  # h/b
sides = 4 if heating == "4 стороны" else 3
lpf_mm = lpf_m * 1000

# Номограммы Приложения В выбираются ПО СХЕМЕ ОБОГРЕВА:
#   4 стороны: Условие I -> ηW4, Условие II -> ηA4
#   3 стороны: Условие I -> ηW3, Условие II -> ηA3
# Раньше в подсказках не учитывалась схема обогрева, из-за этого значение
# могли считывать не с того графика.
graph_w = "ηW4" if sides == 4 else "ηW3"
graph_a = "ηA4" if sides == 4 else "ηA3"

st.title("🔥 Предел огнестойкости клееной деревянной балки (БКП)")
# st.caption(
#    "Сивенков А.Б. и др. «Огнестойкость деревянных конструкций», "
#    "Академия ГПС МЧС России, 2023 — Пример расчёта №1"
# )

st.info(f"📌 Для выбранной схемы **{heating}** используйте графики: Условие I — **{graph_w}**, Условие II — **{graph_a}**.")

# ══════════════════════════════════════════════════════════════════════════════
#  3.1.1 — СБОР НАГРУЗОК, ОПРЕДЕЛЕНИЕ УСИЛИЙ
# ══════════════════════════════════════════════════════════════════════════════
#   qn = q*·a / γf            (кН/м)
#   Mn = qn·L² / 8             (кН·м)
#   Qn = qn·L / 2              (кН)

qn = q_star * a_m / gamma_f
Mn = qn * L_m ** 2 / 8
Qn = qn * L_m / 2

# ══════════════════════════════════════════════════════════════════════════════
#  3.1.2 — ОПРЕДЕЛЕНИЕ ГЕОМЕТРИЧЕСКИХ ХАРАКТЕРИСТИК
# ══════════════════════════════════════════════════════════════════════════════
#   W = b·h² / 6   (м³)
#   A = b·h        (м²)

b, h = b_mm / 1000, h_mm / 1000
W = b * h ** 2 / 6
A = b * h

# ══════════════════════════════════════════════════════════════════════════════
#  3.1.3 — УСЛОВИЕ I: ПРОЧНОСТЬ ПО НОРМАЛЬНЫМ НАПРЯЖЕНИЯМ (ИЗГИБ)
# ══════════════════════════════════════════════════════════════════════════════
#   ηw = Mn / (W·Rfw)
#   По графику Прил. В: 4 стороны -> ηW4, 3 стороны -> ηW3; (ηw, h/b) → Zcr/h
#   Если точка ниже штрихпунктирной линии → Zcr = 0.25·b
#
#   ВАЖНО: номограмма Приложения В — эмпирическая (получена из опытных
#   данных), физическая формула Wf/W её не воспроизводит точно. Поэтому
#   Zcr/h нужно считывать ВРУЧНУЮ с графика номограммы и вводить ниже.

eta_w = (Mn * 1e3) / (W * Rfw * 1e6)
p_dash_w = 0.25 / r          # штрихпунктирная линия по оси Z/h
Z_dash_w = p_dash_w * h_mm   # т.е. Zcr=0.25b в мм

# ══════════════════════════════════════════════════════════════════════════════
#  3.1.4 — УСЛОВИЕ II: ПРОЧНОСТЬ ПО КАСАТЕЛЬНЫМ НАПРЯЖЕНИЯМ (СКАЛЫВАНИЕ)
# ══════════════════════════════════════════════════════════════════════════════
#   ηA = 1.5·Qn / (A·Rfqs)
#   По графику Прил. В: 4 стороны -> ηA4, 3 стороны -> ηA3; (ηA, h/b) → Zcr/h
#   ВАЖНО: значение Zcr/h считывается ВРУЧНУЮ с номограммы (см. ниже).

eta_A = (1.5 * Qn * 1e3) / (A * Rfqs * 1e6)
p_dash_a = 0.25 / r
Z_dash_a = p_dash_a * h_mm

# ── РУЧНОЙ ВВОД Zcr/h ПО НОМОГРАММЕ (Приложение В) ───────────────────────────
#   Номограмма Прил. В — эмпирическая (получена из испытаний), точной
#   аналитической формулы Zcr/h = f(η, h/b) не существует. Поэтому
#   значение Zcr/h считывается ВРУЧНУЮ с графика и вводится здесь —
#   до вычисления Условия III и итоговых метрик.

cola, colb = st.columns(2)

with cola:
    st.subheader("3.1.3 — Условие I: нормальные напряжения")
    st.latex(
        r"\eta_w = \frac{M_n}{W \cdot R_{fw}} = "
        r"\frac{%.2f \times 10^3}{%.6f \times %.0f \times 10^6} = \mathbf{%.4f}"
        % (Mn, W, Rfw, eta_w)
    )
    st.write(f"h/b = **{r:.2f}**")
    st.caption(
        f"📊 Нужный график: **{graph_w}** ({heating}). Найдите кривую h/b={r:.2f}, "
        f"по оси ηw={eta_w:.4f} определите Zcr/h.  \n"
        f"Штрихпунктирная линия: Zcr/h = {p_dash_w:.4f} "
        f"(соответствует Zcr=0.25·b={Z_dash_w:.0f}мм)."
    )

    use_dash_w = st.checkbox(
        "Точка ниже штрихпунктирной линии → Zcr=0.25·b",
        value=False, key="dash_w",
    )
    if use_dash_w:
        Zcr_I = 0.25 * b_mm
        Zcr_I_note = f"Ниже штрихпунктира → Z'cr = 0.25·b = {Zcr_I:.0f} мм"
    else:
        zh_input_I = st.number_input(
            f"Zcr/h (считано с графика {graph_w})",
            min_value=0.0, max_value=0.30, value=0.020, step=0.001,
            format="%.4f", key="zh_I",
        )
        Zcr_I = zh_input_I * h_mm
        Zcr_I_note = f"Zcr/h={zh_input_I:.4f} (ручной ввод) → Z'cr = {Zcr_I:.1f} мм"
    st.info(f"**Zcr I = {Zcr_I:.1f} мм**")

with colb:
    st.subheader("3.1.4 — Условие II: касательные напряжения")
    st.latex(
        r"\eta_A = \frac{1.5\,Q_n}{A \cdot R_{fqs}} = "
        r"\frac{1.5 \times %.2f \times 10^3}{%.5f \times %.1f \times 10^6} = \mathbf{%.4f}"
        % (Qn, A, Rfqs, eta_A)
    )
    st.write(f"h/b = **{r:.2f}**")
    st.caption(
        f"📊 Нужный график: **{graph_a}** ({heating}). Найдите кривую h/b={r:.2f}, "
        f"по оси ηA={eta_A:.4f} определите Zcr/h.  \n"
        f"Штрихпунктирная линия: Zcr/h = {p_dash_a:.4f} "
        f"(соответствует Zcr=0.25·b={Z_dash_a:.0f}мм)."
    )

    use_dash_a = st.checkbox(
        "Точка ниже штрихпунктирной линии → Zcr=0.25·b",
        value=False, key="dash_a",
    )
    if use_dash_a:
        Zcr_II = 0.25 * b_mm
        Zcr_II_note = f"Ниже штрихпунктира → Z''cr = 0.25·b = {Zcr_II:.0f} мм"
    else:
        zh_input_II = st.number_input(
            f"Zcr/h (считано с графика {graph_a})",
            min_value=0.0, max_value=0.30, value=0.015, step=0.001,
            format="%.4f", key="zh_II",
        )
        Zcr_II = zh_input_II * h_mm
        Zcr_II_note = f"Zcr/h={zh_input_II:.4f} (ручной ввод) → Z''cr = {Zcr_II:.1f} мм"
    st.info(f"**Zcr II = {Zcr_II:.1f} мм**")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  3.1.5 — УСЛОВИЕ III: УСТОЙЧИВОСТЬ ПЛОСКОЙ ФОРМЫ ДЕФОРМИРОВАНИЯ
# ══════════════════════════════════════════════════════════════════════════════
#   Zcr1 = (τос-τ0)·υ        Zcr4 = 0.25·b
#   В пределах Zcr1..Zcr4 берутся 4 точки (Zcr1, Zcr2, Zcr3, Zcr4)
#
#   φfM(i) = 250·(b - k·Z(i))² / ((h - n·Z(i))·lpf) · Kfф · Kfжм
#
#       Kfжм = 1 для элементов с ПОСТОЯННОЙ высотой сечения (БКП)
#
#       если lpf < 0.5L:  αf = M(lpf)/Mn,  Kfф = 1.75 - 0.75·αf
#                         M(lpf) = (qn/2)·(L/2-lpf)·(L/2+lpf)
#       если lpf = 0.5L:  c = lpf/2,  Kfф = 1.35 + 1.45·(c/lpf)²
#
#   σfw(i) = Mn / (φfM(i) · W · ηw(i))
#   Графически находят Zcr из условия σfw(Zcr) = Rfw

Zcr1 = (tau_os - tau_0) * v
Zcr4 = 0.25 * b_mm

Kfjm = 1.0  # постоянная высота сечения
half_L = L_m / 2

if lpf_m < half_L - 0.001:
    M_lpf_kNm = (qn / 2) * (L_m / 2 - lpf_m) * (L_m / 2 + lpf_m)
    alpha_f = M_lpf_kNm / Mn
    Kff = 1.75 - 0.75 * alpha_f
    Kff_note = (
        f"lpf < 0.5L: αf = M(lpf)/Mn = {M_lpf_kNm:.2f}/{Mn:.2f} = {alpha_f:.4f}  →  "
        f"Kfф = 1.75 - 0.75·{alpha_f:.4f} = {Kff:.4f}"
    )
else:
    c_m = lpf_m / 2
    Kff = 1.35 + 1.45 * (c_m / lpf_m) ** 2
    Kff_note = (
        f"lpf = 0.5L: c = lpf/2 = {c_m:.2f} м  →  "
        f"Kfф = 1.35 + 1.45·({c_m:.2f}/{lpf_m:.2f})² = {Kff:.4f}"
    )

Kfm_total = Kff * Kfjm

if Zcr4 > Zcr1:
    step = (Zcr4 - Zcr1) / 3
    Z_pts = [Zcr1, Zcr1 + step, Zcr1 + 2 * step, Zcr4]
else:
    Z_pts = [Zcr1] * 4


def phi_fM(Z_mm: float) -> float:
    """φfM = 250·(b-kZ)²/((h-nZ)·lpf)·Kfф·Kfжм  [безразмерный]"""
    bf_mm = b_mm - k * Z_mm
    hf_mm = h_mm - n * Z_mm
    if bf_mm <= 0 or hf_mm <= 0:
        return 1e-9
    return 250 * bf_mm ** 2 / (hf_mm * lpf_mm) * Kfm_total


def eta_w_at(Z_mm: float) -> float:
    """ηw(Z) = Wf/W = (b-kZ)(h-nZ)²/(b·h²)"""
    bf_mm = b_mm - k * Z_mm
    hf_mm = h_mm - n * Z_mm
    if bf_mm <= 0 or hf_mm <= 0:
        return 0.0
    return (bf_mm / b_mm) * (hf_mm / h_mm) ** 2


def sigma_fw(Z_mm: float) -> float:
    """σfw(Z) = Mn / (φfM·W·ηw)  [МПа]"""
    phi = phi_fM(Z_mm)
    ew = eta_w_at(Z_mm)
    if phi < 1e-8 or ew < 1e-8:
        return 1e9
    return (Mn * 1e3) / (phi * W * ew * 1e6)


sigma_pts = [sigma_fw(z) for z in Z_pts]

Zcr_III = None
III_note = ""
if sigma_pts[0] >= Rfw:
    Zcr_III = Zcr1
    III_note = (
        f"σfw(Zcr1={Zcr1:.1f}мм) = {sigma_pts[0]:.2f} МПа ≥ Rfw={Rfw} → "
        f"Zcr = Zcr1, Пф = τос"
    )
elif sigma_pts[-1] <= Rfw:
    Zcr_III = Zcr4
    III_note = f"σfw никогда не достигает Rfw в диапазоне → Zcr = Zcr4 = {Zcr4:.0f} мм"
else:
    try:
        Zcr_III = brentq(lambda z: sigma_fw(z) - Rfw, Zcr1, Zcr4, xtol=0.01)
        III_note = f"σfw(Z) = Rfw при Zcr = {Zcr_III:.1f} мм (по графику σfw=f(Zcr))"
    except Exception:
        Zcr_III = Zcr4
        III_note = "Корень не найден → Zcr = Zcr4"

# ══════════════════════════════════════════════════════════════════════════════
#  ИТОГ: Zcr,min, Пф, КЛАСС ОГНЕСТОЙКОСТИ
# ══════════════════════════════════════════════════════════════════════════════

cond_vals = {"I": Zcr_I, "II": Zcr_II, "III": Zcr_III}
Zcr_fin   = min(cond_vals.values())
gov       = min(cond_vals, key=cond_vals.get)

if gov == "III" and sigma_pts[0] >= Rfw:
    Pf = float(tau_os)
else:
    Pf = pf_formula(Zcr_fin, tau_0, delta, v)
fc = fire_class(Pf)

# ── Защита: сечение перегружено ────────────────────────────────────────────────
if eta_w >= 1.0:
    parts = [
        f"Сечение не несёт нагрузку до пожара (ηw={eta_w:.3f} ≥ 1.0).",
        "Увеличьте h, уменьшите q* или проверьте ориентацию сечения.",
    ]
    if b_mm > h_mm:
        parts.append(
            f"Подсказка: возможно b и h перепутаны. "
            f"Попробуйте b={int(h_mm)}мм, h={int(b_mm)}мм "
            "(стандартно: b=ширина, h=высота)."
        )
    st.error("  \n".join(parts))
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  МЕТРИКИ
# ══════════════════════════════════════════════════════════════════════════════

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Пф, мин", f"{Pf:.1f}")
c2.metric("Класс", fc)
c3.metric("Zcr, мм", f"{Zcr_fin:.1f}")
c4.metric("υ, мм/мин", f"{v}")
c5.metric("Rfw/Rfqs, МПа", f"{Rfw}/{Rfqs}")
st.error(f"⚠️ Определяющее условие: **{COND[gov]}** — Zcr = {Zcr_fin:.1f} мм")
st.divider()

with st.expander("ℹ️ Принятые характеристики материала", expanded=False):
    ci1, ci2, ci3 = st.columns(3)
    with ci1:
        st.markdown("**Сопротивления (Табл. 1.1)**")
        st.write(f"Rfw = **{Rfw} МПа** — изгиб, сорт {sort}")
        st.write(f"Rfqs = **{Rfqs} МПа** — скалывание, {material}")
    with ci2:
        st.markdown("**Обугливание (Табл. 1.2)**")
        st.write(f"υ = **{v} мм/мин**")
        st.write(f"δ = **{delta} мм** (прогретый слой)")
        st.write(f"τ₀ = **{tau_0} мин**")
    with ci3:
        st.markdown("**Устойчивость (Прил. Д)**")
        st.write(f"k = **{k}**, n = **{n}** ({heating})")
        st.write(f"Kfф = **{Kff:.4f}**, Kfжм = **{Kfjm}**")
        st.write(f"lpf = **{lpf_mm:.0f} мм**")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  ПОПЕРЕЧНОЕ СЕЧЕНИЕ — k=2,n=1 (3 стороны, П-форма) vs k=2,n=2 (4 стороны)
# ══════════════════════════════════════════════════════════════════════════════
#   Слайд 33: при k=2,n=1 верхняя грань (h) защищена, обугливание идёт
#   с боков (k=2 раза по b) и снизу (n=1 раз по h) → форма «П» (U).
#   При k=2,n=2 обугливание идёт со всех 4 сторон равномерно (прямоугольник).

st.subheader("Поперечное сечение в расчётный момент Zcr")
st.caption(
    f"k={k}, n={n} ({heating}) — геометрия остаточного сечения "
    f"при Z = Zcr = {Zcr_fin:.1f} мм"
)

Zc = Zcr_fin
bf_mm = max(b_mm - k * Zc, 0.0)
hf_mm = max(h_mm - n * Zc, 0.0)

fig_sec = go.Figure()

# Исходный контур (до пожара)
fig_sec.add_shape(
    type="rect", x0=0, y0=0, x1=b_mm, y1=h_mm,
    line=dict(color="#999", width=1.5, dash="dot"),
    fillcolor="rgba(0,0,0,0)",
)
# Обугленная зона — целиком серая подложка
fig_sec.add_shape(type="rect", x0=0, y0=0, x1=b_mm, y1=h_mm,
                   fillcolor="rgba(120,120,120,0.30)", line=dict(width=0))

work_color = "#A0793D"
edge_color = "#D4A017"
wall_color = "#6B4423"
delta_mm = max(min(b_mm, h_mm) * 0.12, 8)  # стенка короба для наглядности

if n == 1:
    # 3 СТОРОНЫ: верхняя грань (y=h) защищена — П (U) форма короба.
    # Открытый верх, стенки толщиной delta слева/справа/снизу.
    x_left, x_right = Zc, b_mm - Zc
    y_bottom, y_top = Zc, h_mm

    # Внешний контур рабочего сечения (полная зона до обугливания) — фон
    fig_sec.add_shape(type="rect", x0=x_left, y0=y_bottom, x1=x_right, y1=y_top,
                       fillcolor=work_color, line=dict(color=edge_color, width=2))

    # Внутренняя полость короба (П-форма): вырезаем сверху открытую часть
    # Левая стенка, правая стенка, нижняя стенка толщиной delta_mm
    inner_x_left  = x_left + delta_mm
    inner_x_right = x_right - delta_mm
    inner_y_bottom = y_bottom + delta_mm
    if inner_x_left < inner_x_right and inner_y_bottom < y_top:
        fig_sec.add_shape(
            type="rect", x0=inner_x_left, y0=inner_y_bottom,
            x1=inner_x_right, y1=y_top + 5,  # +5 чтобы "вырез" доходил до верха (открыт)
            fillcolor="white", line=dict(color=wall_color, width=1.5))

    fig_sec.add_annotation(
        x=(x_left + x_right) / 2, y=y_bottom + (y_top - y_bottom) * 0.15,
        text=f"<b>{bf_mm:.0f}×{hf_mm:.0f} мм</b><br>рабочее сечение (П-форма)",
        showarrow=False, font=dict(color="#7B4F00", size=12))
    fig_sec.add_annotation(
        x=b_mm / 2, y=h_mm + 16,
        text="верх (грань h) защищён — не обугливается",
        showarrow=False, font=dict(color="#555", size=10))
else:
    # 4 СТОРОНЫ: замкнутый короб — стенки со всех сторон, обугливание равномерное
    x_left, x_right = Zc, b_mm - Zc
    y_bottom, y_top = Zc, h_mm - Zc

    fig_sec.add_shape(type="rect", x0=x_left, y0=y_bottom, x1=x_right, y1=y_top,
                       fillcolor=work_color, line=dict(color=edge_color, width=2))

    inner_x_left  = x_left + delta_mm
    inner_x_right = x_right - delta_mm
    inner_y_bottom = y_bottom + delta_mm
    inner_y_top    = y_top - delta_mm
    if inner_x_left < inner_x_right and inner_y_bottom < inner_y_top:
        fig_sec.add_shape(
            type="rect", x0=inner_x_left, y0=inner_y_bottom,
            x1=inner_x_right, y1=inner_y_top,
            fillcolor="white", line=dict(color=wall_color, width=1.5))

    fig_sec.add_annotation(
        x=(x_left + x_right) / 2, y=(y_bottom + y_top) / 2,
        text=f"<b>{bf_mm:.0f}×{hf_mm:.0f} мм</b><br>рабочее сечение (короб)",
        showarrow=False, font=dict(color="#7B4F00", size=12))

# Размерные подписи
fig_sec.add_annotation(x=b_mm + 8, y=h_mm / 2,
                       text=f"h={h_mm}мм", showarrow=False,
                       font=dict(color="#333", size=11), xanchor="left")
fig_sec.add_annotation(x=b_mm / 2, y=-22,
                       text=f"b={b_mm}мм", showarrow=False,
                       font=dict(color="#333", size=11))
fig_sec.add_annotation(x=2, y=max(Zc / 2, 6),
                       text=f"Z={Zc:.1f}мм", showarrow=False,
                       font=dict(color="#CC3300", size=10), xanchor="left")

layout_sec = white_layout(420, title=f"k={k}, n={n}  ({heating})")
layout_sec["xaxis"].update(range=[-15, b_mm + 75], showgrid=False,
                           zeroline=False, showticklabels=False)
layout_sec["yaxis"].update(range=[-38, h_mm + 38], showgrid=False,
                           zeroline=False, showticklabels=False,
                           scaleanchor="x", scaleratio=1)
fig_sec.update_layout(**layout_sec)
st.plotly_chart(fig_sec, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  ГРАФИК — Условие III: σfw = f(Z)
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("3.1.5 — Условие III: устойчивость плоской формы деформирования")
st.caption(
    f"φfM = 250·(b−{k}·Z)² / ((h−{n}·Z)·{lpf_mm:.0f}) · {Kfm_total:.3f}   "
    f"(k={k}, n={n}, lpf={lpf_mm:.0f}мм, Kfф·Kfжм={Kfm_total:.4f})"
)

Z_line = np.linspace(Zcr1, Zcr4, 300)
sig_line = np.array([sigma_fw(z) for z in Z_line])
valid3 = (sig_line > 0) & (sig_line < 500)

fig3 = go.Figure()
fig3.add_hrect(y0=0, y1=Rfw, fillcolor="rgba(76,175,80,0.07)",
               line_width=0, annotation_text="Аюулгүй бүс",
               annotation_font_color="#2E7D32", annotation_position="top left")

if valid3.any():
    fig3.add_trace(go.Scatter(
        x=Z_line[valid3], y=sig_line[valid3],
        mode="lines", name="σfw(Z)", line=dict(color="#1565C0", width=3)))

fig3.add_trace(go.Scatter(
    x=Z_pts, y=sigma_pts, mode="markers+text", name="Тооцоот цэгүүд",
    marker=dict(color="#E65100", size=10, symbol="circle"),
    text=[f"σ={s:.1f}" for s in sigma_pts],
    textposition="top right", textfont=dict(size=11, color="#E65100")))

fig3.add_hline(y=Rfw, line=dict(color="#C62828", dash="dash", width=2),
               annotation_text=f"Rfw = {Rfw} МПа",
               annotation_font_color="#C62828", annotation_font_size=13)
fig3.add_vline(x=Zcr1, line=dict(color="#888", dash="dot", width=1.5),
               annotation_text=f"Zcr1={Zcr1:.1f}мм", annotation_font_color="#888")
fig3.add_vline(x=Zcr4, line=dict(color="#888", dash="dot", width=1.5),
               annotation_text=f"Zcr4={Zcr4:.0f}мм", annotation_font_color="#888")
fig3.add_vline(x=Zcr_III, line=dict(color="#E65100", width=2.5),
               annotation_text=f"Zcr={Zcr_III:.1f}мм",
               annotation_font_color="#E65100", annotation_font_size=13)
if Zcr1 < Zcr_III < Zcr4:
    fig3.add_trace(go.Scatter(x=[Zcr_III], y=[Rfw], mode="markers",
                              marker=dict(color="#C62828", size=14, symbol="x"),
                              name=f"Zcr III={Zcr_III:.1f}мм"))

y_top = min(float(sig_line[valid3].max() * 1.15), 60) if valid3.any() else 35
fig3.update_layout(**white_layout(380, xtitle="Z, мм", ytitle="σfw, МПа"))
fig3.update_yaxes(range=[0, y_top])
st.plotly_chart(fig3, use_container_width=True)

col_a, col_b, col_c, col_d = st.columns(4)
for i, (z, sig) in enumerate(zip(Z_pts, sigma_pts)):
    cols = [col_a, col_b, col_c, col_d]
    cols[i].metric(f"Z{i+1}={z:.1f}мм", f"σ={sig:.2f}МПа",
                   delta=f"φ={phi_fM(z):.3f}")
st.info(f"**Zcr III = {Zcr_III:.1f} мм** — {III_note}")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  СВОДНАЯ ТАБЛИЦА
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("📊 Итог по трём условиям")

rows = []
for cid, Zcr_c in cond_vals.items():
    if cid == "III" and sigma_pts[0] >= Rfw:
        pf_c = float(tau_os)
    else:
        pf_c = pf_formula(Zcr_c, tau_0, delta, v)
    rows.append({
        "Условие":     COND[cid],
        "Zcr, мм":     f"{Zcr_c:.2f}",
        "Пф, мин":     f"{pf_c:.1f}",
        "Класс":       fire_class(pf_c),
        "Определяет":  "✅ ДА" if Zcr_c == Zcr_fin else "—",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ПОШАГОВОЕ РЕШЕНИЕ (точно по нумерации книги: 3.1.1 → 3.1.5)
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("📋 Пошаговое решение (точно по книге, формулы)", expanded=False):

    st.markdown("### 3.1.1. Сбор нагрузок, определение усилий")
    st.latex(
        r"q_n = \frac{q^* \cdot a}{\gamma_f} = "
        r"\frac{%.2f \cdot %.2f}{%.2f} = %.3f \text{ кН/м}"
        % (q_star, a_m, gamma_f, qn)
    )
    st.latex(
        r"M_n = \frac{q_n L^2}{8} = "
        r"\frac{%.3f \cdot %.1f^2}{8} = %.2f \text{ кН·м}" % (qn, L_m, Mn)
    )
    st.latex(
        r"Q_n = \frac{q_n L}{2} = "
        r"\frac{%.3f \cdot %.1f}{2} = %.2f \text{ кН}" % (qn, L_m, Qn)
    )

    st.markdown("### 3.1.2. Определение геометрических характеристик")
    st.latex(
        r"W_x = \frac{b \cdot h^2}{6} = "
        r"\frac{%.3f \cdot %.3f^2}{6} = %.6f \text{ м}^3" % (b, h, W)
    )
    st.latex(r"A = b \cdot h = %.3f \cdot %.3f = %.5f \text{ м}^2" % (b, h, A))
    st.write(f"h/b = {h_mm}/{b_mm} = **{r:.2f}**")

    st.markdown("### 3.1.3. Условие I — прочность по нормальным напряжениям")
    st.latex(
        r"\eta_w = \frac{M_n}{W_x \cdot R_{fw}} = "
        r"\frac{%.2f \times 10^3}{%.6f \times %.0f \times 10^6} = %.4f"
        % (Mn, W, Rfw, eta_w)
    )
    st.write(f"Rfw = **{Rfw} МПа** (табл. 1.1)")
    st.success(f"**Z'cr = {Zcr_I:.1f} мм** — {Zcr_I_note}")

    st.markdown("### 3.1.4. Условие II — прочность по касательным напряжениям")
    st.latex(
        r"\eta_A = \frac{1.5\,Q_n}{A \cdot R_{fqs}} = "
        r"\frac{1.5 \times %.2f \times 10^3}{%.5f \times %.1f \times 10^6} = %.4f"
        % (Qn, A, Rfqs, eta_A)
    )
    st.write(f"Rfqs = **{Rfqs} МПа** (табл. 1.1)")
    st.success(f"**Z''cr = {Zcr_II:.1f} мм** — {Zcr_II_note}")

    st.markdown("### 3.1.5. Условие III — устойчивость плоской формы")
    st.latex(
        r"Z_{cr1} = (\tau_{oc} - \tau_0)\cdot\upsilon = "
        r"(%d - %d)\times %.1f = %.1f \text{ мм}" % (tau_os, tau_0, v, Zcr1)
    )
    st.latex(r"Z_{cr4} = 0.25\,b = 0.25\times%d = %.1f \text{ мм}" % (b_mm, Zcr4))
    st.write(f"υ = **{v} мм/мин** (табл. 1.2)")
    st.write(
        f"Произвольно задаёмся: Zcr2={Z_pts[1]:.1f}мм, Zcr3={Z_pts[2]:.1f}мм "
        f"(в пределах Zcr1..Zcr4)"
    )

    st.markdown(f"**Коэффициент Kfф:** {Kff_note}")
    st.latex(r"K_{f\phi}\cdot K_{f\text{жм}} = %.4f \times %.1f = %.4f"
             % (Kff, Kfjm, Kfm_total))

    st.latex(
        r"\varphi_{fM(i)} = 250\,\frac{(b-k Z_i)^2}{(h-n Z_i)\,l_{pf}}\,"
        r"K_{f\phi}K_{f\text{жм}} \qquad (k=%d,\;n=%d)" % (k, n)
    )
    st.latex(r"\sigma_{fw(i)} = \frac{M_n}{\varphi_{fM(i)}\,W\,\eta_{w(i)}}\quad[\text{МПа}]")

    df_pts = pd.DataFrame({
        "i":          [1, 2, 3, 4],
        "Z, мм":      [f"{z:.1f}" for z in Z_pts],
        "Z/h":        [f"{z/h_mm:.4f}" for z in Z_pts],
        "ηw=Wf/W":    [f"{eta_w_at(z):.4f}" for z in Z_pts],
        "φfM":        [f"{phi_fM(z):.3f}" for z in Z_pts],
        "σfw, МПа":   [f"{s:.2f}" for s in sigma_pts],
    })
    st.dataframe(df_pts, use_container_width=True, hide_index=True)
    st.write(f"По графику σfw=f(Zcr) при Rfw={Rfw}МПа: **{III_note}**")

    st.markdown("### Итог: предел огнестойкости конструкции")
    st.latex(
        r"Z_{cr}^{min} = \min\!\left(%.1f;\;%.1f;\;%.1f\right) = "
        r"\mathbf{%.1f}\text{ мм}" % (Zcr_I, Zcr_II, Zcr_III, Zcr_fin)
    )
    if gov == "III" and sigma_pts[0] >= Rfw:
        st.latex(r"\Pi_\phi = \tau_{oc} = \mathbf{%.0f \text{ мин}}" % tau_os)
    elif Zcr_fin > delta:
        st.latex(
            r"\Pi_\phi = \tau_0 + \frac{Z_{cr} - \delta}{\upsilon} = "
            r"%d + \frac{%.1f - %d}{%.1f} = \mathbf{%.1f \text{ мин}}"
            % (tau_0, Zcr_fin, delta, v, Pf)
        )
    else:
        st.latex(
            r"\Pi_\phi = Z_{cr}\cdot\frac{\tau_0}{\delta} = "
            r"%.1f \cdot \frac{%d}{%d} = \mathbf{%.1f \text{ мин}}"
            % (Zcr_fin, tau_0, delta, Pf)
        )
    st.success(
        f"**Вывод:** фактический предел огнестойкости конструкции равен "
        f"**{fc}** ({Pf:.1f} мин). Потеря несущей способности при пожаре "
        f"наступает из условия: **{COND[gov]}**."
    )

st.divider()
st.caption(
    "© Расчёт по: Сивенков А.Б. и др. «Огнестойкость деревянных конструкций», "
    "Академия ГПС МЧС России, 2023.  \n"
    "Дараалал: 3.1.1 нагрузки → 3.1.2 геометрия → 3.1.3 изгиб → "
    "3.1.4 скалывание → 3.1.5 устойчивость → итог."
)
