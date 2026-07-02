"""
Предел огнестойкости клееной деревянной балки (БКП)
======================================================
Методика: Сивенков А.Б. и др. «Огнестойкость деревянных конструкций»,
Академия ГПС МЧС России, 2023 — Пример расчёта №1 (стр. 73-76)
"""

import math
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import brentq

st.set_page_config(page_title="Огнестойкость БКП / БКП / GLT Fire", page_icon="🔥", layout="wide")
st.markdown(
    "<style>[data-testid='stSidebar']{min-width:360px;max-width:420px}</style>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
#  MULTILINGUAL DICTIONARY
# ══════════════════════════════════════════════════════════════════════════════

T = {
    # --- Language selector ---
    "lang_label": {
        "mn": "Хэл сонгох",
        "ru": "Язык",
        "en": "Language",
    },
    # --- Page title & caption ---
    "page_title": {
        "mn": "🔥 БКП (Холбоосон модон дам нуруу) — Галын тэсвэрлэх чадвар",
        "ru": "🔥 Предел огнестойкости клееной деревянной балки (БКП)",
        "en": "🔥 Fire Resistance of Glued Laminated Timber Beam (GLT)",
    },
    "page_caption": {
        "mn": "Сивенков А.Б. нар «Модон байгууламжийн галын тэсвэрлэх чадвар», ОХУ ТЯЯ ГПС Академи, 2023 — Тооцооны жишээ №1",
        "ru": "Сивенков А.Б. и др. «Огнестойкость деревянных конструкций», Академия ГПС МЧС России, 2023 — Пример расчёта №1",
        "en": "Sivenkov A.B. et al. «Fire Resistance of Timber Structures», Academy of State Fire Service of EMERCOM, 2023 — Worked Example No. 1",
    },
    # --- Sidebar ---
    "sidebar_title": {
        "mn": "⚙️ Анхны өгөгдөл",
        "ru": "⚙️ Исходные данные",
        "en": "⚙️ Input Data",
    },
    "heating_label": {
        "mn": "Халаалтын схем",
        "ru": "Схема обогрева",
        "en": "Heating scheme",
    },
    "heating_4": {
        "mn": "4 тал",
        "ru": "4 стороны",
        "en": "4 sides",
    },
    "heating_3": {
        "mn": "3 тал",
        "ru": "3 стороны",
        "en": "3 sides",
    },
    "geom_header": {
        "mn": "📐 Огтлолцлын геометр",
        "ru": "📐 Геометрия сечения",
        "en": "📐 Cross-section Geometry",
    },
    "b_label": {
        "mn": "b — огтлолцлын өргөн, мм",
        "ru": "b — ширина сечения, мм",
        "en": "b — section width, mm",
    },
    "h_label": {
        "mn": "h — огтлолцлын өндөр, мм",
        "ru": "h — высота сечения, мм",
        "en": "h — section height, mm",
    },
    "span_header": {
        "mn": "📏 Урт параметрүүд",
        "ru": "📏 Параметры пролёта",
        "en": "📏 Span Parameters",
    },
    "L_label": {
        "mn": "L — тооцооны урт, м",
        "ru": "L — расчётный пролёт, м",
        "en": "L — design span, m",
    },
    "a_label": {
        "mn": "a — дам нурууны алхам, м",
        "ru": "a — шаг балок, м",
        "en": "a — beam spacing, m",
    },
    "lp_label": {
        "mn": "lp — шахалтын ирмэгийг бэхлэх урт, м",
        "ru": "lp — закрепление сжатой кромки, м",
        "en": "lp — compressed edge bracing, m",
    },
    "lpf_label": {
        "mn": "lpf — гал гарсан үед холбоосгүй хэсгийн урт, м",
        "ru": "lpf — длина участка без связей при пожаре, м",
        "en": "lpf — unbraced length during fire, m",
    },
    "load_header": {
        "mn": "📦 Ачаалал",
        "ru": "📦 Нагрузка",
        "en": "📦 Load",
    },
    "q_star_label": {
        "mn": "q* — тооцооны ачаалал, кН/м²",
        "ru": "q* — расчётная нагрузка, кН/м²",
        "en": "q* — design load, kN/m²",
    },
    "gamma_f_label": {
        "mn": "γf — ачааллын найдвартай коэффициент",
        "ru": "γf — коэф. надёжности по нагрузке",
        "en": "γf — load safety factor",
    },
    "material_header": {
        "mn": "🪵 Материал",
        "ru": "🪵 Материал",
        "en": "🪵 Material",
    },
    "sort_label": {
        "mn": "Модны ангилал",
        "ru": "Сорт древесины",
        "en": "Timber grade",
    },
    "material_label": {
        "mn": "Материалын төрөл",
        "ru": "Тип материала",
        "en": "Material type",
    },
    "glued": {
        "mn": "Холбоосон",
        "ru": "Клееная",
        "en": "Glued laminated",
    },
    "solid": {
        "mn": "Цул",
        "ru": "Цельная",
        "en": "Solid",
    },
    "material_note": {
        "mn": "💡 Номын жишээ №1 (73–76 хуудас): «цул» гэж заасан боловч Нөхцөл II-д Rfqs нь холбоосны (1.2 МПа) утгыг авсан — эх бичигт зөрчил байна.",
        "ru": "💡 Пример №1 книги (стр. 73–76): материал указан как «цельная», но Rfqs в расчёте Условия II взято для клееной (1.2 МПа) — несоответствие в тексте оригинала.",
        "en": "💡 Book Example No. 1 (pp. 73–76): material listed as 'solid', but Rfqs in Condition II uses glued value (1.2 MPa) — inconsistency in original text.",
    },
    "char_header": {
        "mn": "🔥 Нүүрслэлт",
        "ru": "🔥 Обугливание",
        "en": "🔥 Charring",
    },
    "tau0_label": {
        "mn": "τ₀ — нүүрслэлт эхлэх хүртэлх хугацаа, мин",
        "ru": "τ₀ — время до начала обугливания, мин",
        "en": "τ₀ — time to start of charring, min",
    },
    "delta_label": {
        "mn": "δ — халсан давхаргын зузаан, мм",
        "ru": "δ — толщина прогретого слоя, мм",
        "en": "δ — heated layer thickness, mm",
    },
    "tau_os_label": {
        "mn": "τос — холбоосны галын тэсвэрлэх чадварын хязгаар, мин",
        "ru": "τос — предел огнестойкости связей, мин",
        "en": "τос — fire resistance of bracings, min",
    },
    "sidebar_ref": {
        "mn": "Сивенков А.Б. нар\n«Модон байгууламжийн галын тэсвэрлэх чадвар»\nОХУ ТЯЯ ГПС Академи, 2023",
        "ru": "Сивенков А.Б. и др.\n«Огнестойкость деревянных конструкций»\nАкадемия ГПС МЧС России, 2023",
        "en": "Sivenkov A.B. et al.\n«Fire Resistance of Timber Structures»\nAcademy of State Fire Service, 2023",
    },
    # --- Condition names ---
    "cond_I": {
        "mn": "Нормал хүчдэлийн нөхцөл (нугалалт)",
        "ru": "Нормальные напряжения (изгиб)",
        "en": "Normal stresses (bending)",
    },
    "cond_II": {
        "mn": "Тангенциал хүчдэлийн нөхцөл (хуулалт)",
        "ru": "Касательные напряжения (скалывание)",
        "en": "Shear stresses",
    },
    "cond_III": {
        "mn": "Хавтгай деформацийн тогтворт байдал",
        "ru": "Устойчивость плоской формы",
        "en": "Lateral-torsional stability",
    },
    # --- Metrics ---
    "pf_label": {
        "mn": "Пф, мин",
        "ru": "Пф, мин",
        "en": "FRL, min",
    },
    "class_label": {
        "mn": "Анги",
        "ru": "Класс",
        "en": "Class",
    },
    "zcr_label": {
        "mn": "Zcr, мм",
        "ru": "Zcr, мм",
        "en": "Zcr, mm",
    },
    "v_label": {
        "mn": "υ, мм/мин",
        "ru": "υ, мм/мин",
        "en": "υ, mm/min",
    },
    "governing_cond": {
        "mn": "⚠️ Тодорхойлох нөхцөл: **{cond}** — Zcr = {z:.1f} мм",
        "ru": "⚠️ Определяющее условие: **{cond}** — Zcr = {z:.1f} мм",
        "en": "⚠️ Governing condition: **{cond}** — Zcr = {z:.1f} mm",
    },
    # --- Section geometry ---
    "sec_subheader": {
        "mn": "Тооцоот мөчид огтлол (Zcr)",
        "ru": "Поперечное сечение в расчётный момент Zcr",
        "en": "Cross-section at design moment Zcr",
    },
    "sec_caption": {
        "mn": "k={k}, n={n} ({heating}) — Z = Zcr = {z:.1f} мм дэх үлдэгдэл огтлол",
        "ru": "k={k}, n={n} ({heating}) — геометрия остаточного сечения при Z = Zcr = {z:.1f} мм",
        "en": "k={k}, n={n} ({heating}) — residual section at Z = Zcr = {z:.1f} mm",
    },
    "protected_top": {
        "mn": "дээд ирмэг (h тал) хамгаалагдсан — нүүрсдэхгүй",
        "ru": "верх (грань h) защищён — не обугливается",
        "en": "top face (h side) protected — no charring",
    },
    "working_sec_u": {
        "mn": "ажлын огтлол (П-хэлбэр)",
        "ru": "рабочее сечение (П-форма)",
        "en": "working section (U-shape)",
    },
    "working_sec_box": {
        "mn": "ажлын огтлол (хайрцаг)",
        "ru": "рабочее сечение (короб)",
        "en": "working section (box)",
    },
    "safe_zone": {
        "mn": "Аюулгүй бүс",
        "ru": "Аюулгүй бүс",
        "en": "Safe zone",
    },
    "calc_pts": {
        "mn": "Тооцоот цэгүүд",
        "ru": "Тооцоот цэгүүд",
        "en": "Design points",
    },
    # --- Cond I/II subheaders ---
    "c1_subheader": {
        "mn": "3.1.3 — Нөхцөл I: нормал хүчдэл",
        "ru": "3.1.3 — Условие I: нормальные напряжения",
        "en": "3.1.3 — Condition I: normal stresses",
    },
    "c2_subheader": {
        "mn": "3.1.4 — Нөхцөл II: тангенциал хүчдэл",
        "ru": "3.1.4 — Условие II: касательные напряжения",
        "en": "3.1.4 — Condition II: shear stresses",
    },
    "c1_caption": {
        "mn": "📊 Номограммаас (В хавсралт) h/b={r:.2f} муруйг олж, ηw={eta:.4f} тэнхлэгийн дагуу Zcr/h-г унш.\nШтрихпункт шугам: Zcr/h = {p:.4f} (Zcr=0.25·b={z:.0f}мм).",
        "ru": "📊 По номограмме (Приложение В) найдите кривую h/b={r:.2f}, по оси ηw={eta:.4f} определите Zcr/h.\nШтрихпунктирная линия: Zcr/h = {p:.4f} (соответствует Zcr=0.25·b={z:.0f}мм).",
        "en": "📊 From nomogram (Appendix V), find curve h/b={r:.2f}, read Zcr/h at ηw={eta:.4f}.\nDash-dot line: Zcr/h = {p:.4f} (i.e. Zcr=0.25·b={z:.0f}mm).",
    },
    "c2_caption": {
        "mn": "📊 Номограммаас (В хавсралт) h/b={r:.2f} муруйг олж, ηA={eta:.4f} тэнхлэгийн дагуу Zcr/h-г унш.\nШтрихпункт шугам: Zcr/h = {p:.4f} (Zcr=0.25·b={z:.0f}мм).",
        "ru": "📊 По номограмме (Приложение В) найдите кривую h/b={r:.2f}, по оси ηA={eta:.4f} определите Zcr/h.\nШтрихпунктирная линия: Zcr/h = {p:.4f} (соответствует Zcr=0.25·b={z:.0f}мм).",
        "en": "📊 From nomogram (Appendix V), find curve h/b={r:.2f}, read Zcr/h at ηA={eta:.4f}.\nDash-dot line: Zcr/h = {p:.4f} (i.e. Zcr=0.25·b={z:.0f}mm).",
    },
    "dash_checkbox": {
        "mn": "Цэг штрихпункт шугаанаас доор → Zcr=0.25·b",
        "ru": "Точка ниже штрихпунктирной линии → Zcr=0.25·b",
        "en": "Point below dash-dot line → Zcr=0.25·b",
    },
    "zh_input_label": {
        "mn": "Zcr/h (номограммаас унших)",
        "ru": "Zcr/h (считано с номограммы)",
        "en": "Zcr/h (read from nomogram)",
    },
    "zcr_I_info": {
        "mn": "**Zcr I = {z:.1f} мм**",
        "ru": "**Zcr I = {z:.1f} мм**",
        "en": "**Zcr I = {z:.1f} mm**",
    },
    "zcr_II_info": {
        "mn": "**Zcr II = {z:.1f} мм**",
        "ru": "**Zcr II = {z:.1f} мм**",
        "en": "**Zcr II = {z:.1f} mm**",
    },
    "dash_w_note_below": {
        "mn": "Штрихпунктираас доор → Z'cr = 0.25·b = {z:.0f} мм",
        "ru": "Ниже штрихпунктира → Z'cr = 0.25·b = {z:.0f} мм",
        "en": "Below dash-dot → Z'cr = 0.25·b = {z:.0f} mm",
    },
    "dash_a_note_below": {
        "mn": "Штрихпунктираас доор → Z''cr = 0.25·b = {z:.0f} мм",
        "ru": "Ниже штрихпунктира → Z''cr = 0.25·b = {z:.0f} мм",
        "en": "Below dash-dot → Z''cr = 0.25·b = {z:.0f} mm",
    },
    "zh_I_note": {
        "mn": "Zcr/h={zh:.4f} (гараар) → Z'cr = {z:.1f} мм",
        "ru": "Zcr/h={zh:.4f} (ручной ввод) → Z'cr = {z:.1f} мм",
        "en": "Zcr/h={zh:.4f} (manual) → Z'cr = {z:.1f} mm",
    },
    "zh_II_note": {
        "mn": "Zcr/h={zh:.4f} (гараар) → Z''cr = {z:.1f} мм",
        "ru": "Zcr/h={zh:.4f} (ручной ввод) → Z''cr = {z:.1f} мм",
        "en": "Zcr/h={zh:.4f} (manual) → Z''cr = {z:.1f} mm",
    },
    # --- Cond III ---
    "c3_subheader": {
        "mn": "3.1.5 — Нөхцөл III: хавтгай деформацийн тогтворт байдал",
        "ru": "3.1.5 — Условие III: устойчивость плоской формы деформирования",
        "en": "3.1.5 — Condition III: lateral-torsional stability",
    },
    # --- Summary table ---
    "table_subheader": {
        "mn": "📊 Гурван нөхцөлийн дүн",
        "ru": "📊 Итог по трём условиям",
        "en": "📊 Summary of Three Conditions",
    },
    "col_cond": {
        "mn": "Нөхцөл",
        "ru": "Условие",
        "en": "Condition",
    },
    "col_zcr": {
        "mn": "Zcr, мм",
        "ru": "Zcr, мм",
        "en": "Zcr, mm",
    },
    "col_pf": {
        "mn": "Пф, мин",
        "ru": "Пф, мин",
        "en": "FRL, min",
    },
    "col_class": {
        "mn": "Анги",
        "ru": "Класс",
        "en": "Class",
    },
    "col_governing": {
        "mn": "Тодорхойлно",
        "ru": "Определяет",
        "en": "Governing",
    },
    "yes": {
        "mn": "✅ ТИЙМ",
        "ru": "✅ ДА",
        "en": "✅ YES",
    },
    # --- Expander step-by-step ---
    "expander_label": {
        "mn": "📋 Алхам алхмаар шийдэл (номын дарааллаар: 3.1.1 → 3.1.5)",
        "ru": "📋 Пошаговое решение (точно по книге, формулы)",
        "en": "📋 Step-by-step solution (following book order: 3.1.1 → 3.1.5)",
    },
    "step311": {
        "mn": "### 3.1.1. Ачааллыг нэгтгэх, хүчлэлийг тодорхойлох",
        "ru": "### 3.1.1. Сбор нагрузок, определение усилий",
        "en": "### 3.1.1. Load collection, determination of forces",
    },
    "step312": {
        "mn": "### 3.1.2. Геометрийн шинж чанарыг тодорхойлох",
        "ru": "### 3.1.2. Определение геометрических характеристик",
        "en": "### 3.1.2. Determination of geometric properties",
    },
    "step313": {
        "mn": "### 3.1.3. Нөхцөл I — нормал хүчдэлийн бат бэх",
        "ru": "### 3.1.3. Условие I — прочность по нормальным напряжениям",
        "en": "### 3.1.3. Condition I — strength in normal stresses",
    },
    "step314": {
        "mn": "### 3.1.4. Нөхцөл II — тангенциал хүчдэлийн бат бэх",
        "ru": "### 3.1.4. Условие II — прочность по касательным напряжениям",
        "en": "### 3.1.4. Condition II — shear strength",
    },
    "step315": {
        "mn": "### 3.1.5. Нөхцөл III — хавтгай хэлбэрийн тогтворт байдал",
        "ru": "### 3.1.5. Условие III — устойчивость плоской формы",
        "en": "### 3.1.5. Condition III — lateral-torsional stability",
    },
    "step_result": {
        "mn": "### Дүн: байгууламжийн галын тэсвэрлэх чадварын хязгаар",
        "ru": "### Итог: предел огнестойкости конструкции",
        "en": "### Result: fire resistance limit of the structure",
    },
    "rfw_table": {
        "mn": "Rfw = **{rfw} МПа** (Хүс. 1.1)",
        "ru": "Rfw = **{rfw} МПа** (табл. 1.1)",
        "en": "Rfw = **{rfw} MPa** (Table 1.1)",
    },
    "rfqs_table": {
        "mn": "Rfqs = **{rfqs} МПа** (Хүс. 1.1)",
        "ru": "Rfqs = **{rfqs} МПа** (табл. 1.1)",
        "en": "Rfqs = **{rfqs} MPa** (Table 1.1)",
    },
    "hb_ratio": {
        "mn": "h/b = {hb:.2f}",
        "ru": "h/b = {hb:.2f}",
        "en": "h/b = {hb:.2f}",
    },
    "arb_pts": {
        "mn": "Дурын цэгүүд: Zcr2={z2:.1f}мм, Zcr3={z3:.1f}мм (Zcr1..Zcr4 дотор)",
        "ru": "Произвольно задаёмся: Zcr2={z2:.1f}мм, Zcr3={z3:.1f}мм (в пределах Zcr1..Zcr4)",
        "en": "Chosen intermediate: Zcr2={z2:.1f}mm, Zcr3={z3:.1f}mm (within Zcr1..Zcr4)",
    },
    "v_note": {
        "mn": "υ = **{v} мм/мин** (Хүс. 1.2)",
        "ru": "υ = **{v} мм/мин** (табл. 1.2)",
        "en": "υ = **{v} mm/min** (Table 1.2)",
    },
    "graph_note": {
        "mn": "σfw=f(Zcr) графикаас Rfw={rfw}МПа-д: **{note}**",
        "ru": "По графику σfw=f(Zcr) при Rfw={rfw}МПа: **{note}**",
        "en": "From σfw=f(Zcr) graph at Rfw={rfw}MPa: **{note}**",
    },
    "conclusion": {
        "mn": "**Дүгнэлт:** байгууламжийн галын тэсвэрлэх чадварын бодит хязгаар **{fc}** ({pf:.1f} мин) тэнцэнэ. Гал гарсан үед нөхцөлийн дагуу тулах чадвар алдагдана: **{cond}**.",
        "ru": "**Вывод:** фактический предел огнестойкости конструкции равен **{fc}** ({pf:.1f} мин). Потеря несущей способности при пожаре наступает из условия: **{cond}**.",
        "en": "**Conclusion:** the actual fire resistance limit of the structure is **{fc}** ({pf:.1f} min). Load-bearing capacity is lost in fire according to condition: **{cond}**.",
    },
    "material_props": {
        "mn": "ℹ️ Хүлээн авсан материалын шинж чанар",
        "ru": "ℹ️ Принятые характеристики материала",
        "en": "ℹ️ Adopted material properties",
    },
    "resist_header": {
        "mn": "**Эсэргүүцлүүд (Хүс. 1.1)**",
        "ru": "**Сопротивления (Табл. 1.1)**",
        "en": "**Resistances (Table 1.1)**",
    },
    "char_props": {
        "mn": "**Нүүрслэлт (Хүс. 1.2)**",
        "ru": "**Обугливание (Табл. 1.2)**",
        "en": "**Charring (Table 1.2)**",
    },
    "stab_props": {
        "mn": "**Тогтворт байдал (Д хавсралт)**",
        "ru": "**Устойчивость (Прил. Д)**",
        "en": "**Stability (Appendix D)**",
    },
    "overloaded_err": {
        "mn": "Огтлол галын өмнө ачааллыг дааж чадахгүй байна (ηw={eta:.3f} ≥ 1.0).\nh-г нэмэгдүүлэх, q*-г багасгах эсвэл огтлолын чиглэлийг шалгана уу.",
        "ru": "Сечение не несёт нагрузку до пожара (ηw={eta:.3f} ≥ 1.0).\nУвеличьте h, уменьшите q* или проверьте ориентацию сечения.",
        "en": "Section cannot carry load before fire (ηw={eta:.3f} ≥ 1.0).\nIncrease h, decrease q*, or check section orientation.",
    },
    "bh_swap_hint": {
        "mn": "Зөвлөмж: b болон h солигдсон байж болзошгүй. b={b}мм, h={h}мм туршина уу (стандарт: b=өргөн, h=өндөр).",
        "ru": "Подсказка: возможно b и h перепутаны. Попробуйте b={b}мм, h={h}мм (стандартно: b=ширина, h=высота).",
        "en": "Hint: b and h may be swapped. Try b={b}mm, h={h}mm (standard: b=width, h=height).",
    },
    "footer": {
        "mn": "© Тооцоо: Сивенков А.Б. нар «Модон байгууламжийн галын тэсвэрлэх чадвар», ОХУ ТЯЯ ГПС Академи, 2023.  \nДараалал: 3.1.1 ачаалал → 3.1.2 геометр → 3.1.3 нугалалт → 3.1.4 хуулалт → 3.1.5 тогтворт байдал → дүн.",
        "ru": "© Расчёт по: Сивенков А.Б. и др. «Огнестойкость деревянных конструкций», Академия ГПС МЧС России, 2023.  \nДараалал: 3.1.1 нагрузки → 3.1.2 геометрия → 3.1.3 изгиб → 3.1.4 скалывание → 3.1.5 устойчивость → итог.",
        "en": "© Calculation per: Sivenkov A.B. et al. «Fire Resistance of Timber Structures», Academy of State Fire Service of EMERCOM, 2023.  \nOrder: 3.1.1 loads → 3.1.2 geometry → 3.1.3 bending → 3.1.4 shear → 3.1.5 stability → result.",
    },
    "df_cols": {
        "mn": ["i", "Z, мм", "Z/h", "ηw=Wf/W", "φfM", "σfw, МПа"],
        "ru": ["i", "Z, мм", "Z/h", "ηw=Wf/W", "φfM", "σfw, МПа"],
        "en": ["i", "Z, mm", "Z/h", "ηw=Wf/W", "φfM", "σfw, MPa"],
    },
}


def t(key, lang, **kwargs):
    """Get translation, optionally format with kwargs."""
    val = T[key][lang]
    if kwargs:
        val = val.format(**kwargs)
    return val


# ══════════════════════════════════════════════════════════════════════════════
#  LANGUAGE SELECTOR  (top of sidebar — first widget so session state loads)
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    lang_options = {"🇲🇳 Монгол": "mn", "🇷🇺 Русский": "ru", "🇬🇧 English": "en"}
    lang_display = st.radio(
        "🌐 " + {"mn": "Хэл", "ru": "Язык", "en": "Language"}.get(
            st.session_state.get("lang", "mn"), "Хэл"
        ),
        list(lang_options.keys()),
        horizontal=True,
        key="lang_radio",
    )
    lang = lang_options[lang_display]

# ══════════════════════════════════════════════════════════════════════════════
#  ТАБЛИЦЫ (Табл. 1.1, 1.2, Приложение Д)
# ══════════════════════════════════════════════════════════════════════════════

RFW    = {"1": 29.0, "2": 26.0, "3": 18.0}
RFQS_G = {"1": 1.3,  "2": 1.2,  "3": 1.1}
RFQS_S = {"1": 3.7,  "2": 3.2,  "3": 2.9}
CHAR   = {("Клееная", True): 0.6, ("Клееная", False): 0.7,
          ("Цельная", True): 0.8, ("Цельная", False): 1.0}
KN     = {"4": (2, 2), "3": (2, 1)}

FIRE_CLS = [(15, "R15"), (30, "R30"), (45, "R45"), (60, "R60"),
            (90, "R90"), (120, "R120"), (180, "R180")]


def fire_class(tt: float) -> str:
    for lim, cls in FIRE_CLS:
        if tt <= lim:
            return cls
    return "R180+"


def pf_formula(Z_mm: float, tau0: float, delta_mm: float, vv: float) -> float:
    if Z_mm > delta_mm:
        return tau0 + (Z_mm - delta_mm) / vv
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
#  SIDEBAR — INPUT DATA
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title(t("sidebar_title", lang))

    heating_4_lbl = t("heating_4", lang)
    heating_3_lbl = t("heating_3", lang)
    heating_display = st.radio(
        t("heating_label", lang),
        [heating_4_lbl, heating_3_lbl],
        horizontal=True,
    )
    sides_key = "4" if heating_display == heating_4_lbl else "3"
    k, n = KN[sides_key]
    heating_label_for_display = t("heating_4" if sides_key == "4" else "heating_3", lang)

    st.divider()
    st.subheader(t("geom_header", lang))
    b_mm = st.number_input(t("b_label", lang), 60, 500, 140, 5)
    h_mm = st.number_input(t("h_label", lang), 100, 2000, 710, 10)

    st.subheader(t("span_header", lang))
    L_m   = st.number_input(t("L_label", lang), 1.0, 30.0, 6.0, 0.5)
    a_m   = st.number_input(t("a_label", lang), 0.5, 10.0, 3.0, 0.25)
    lp_m  = st.number_input(t("lp_label", lang), 0.5, 20.0, 1.5, 0.25)
    lpf_m = st.number_input(t("lpf_label", lang), 0.5, 30.0, 3.0, 0.5)

    st.divider()
    st.subheader(t("load_header", lang))
    q_star  = st.number_input(t("q_star_label", lang), 0.5, 50.0, 9.35, 0.05, format="%.2f")
    gamma_f = st.number_input(t("gamma_f_label", lang), 1.0, 2.0, 1.2, 0.05, format="%.2f")

    st.divider()
    st.subheader(t("material_header", lang))
    sort     = st.selectbox(t("sort_label", lang), ["1", "2", "3"], index=1)

    glued_lbl = t("glued", lang)
    solid_lbl = t("solid", lang)
    material_display = st.selectbox(t("material_label", lang), [glued_lbl, solid_lbl])
    material = "Клееная" if material_display == glued_lbl else "Цельная"
    st.caption(t("material_note", lang))

    st.divider()
    st.subheader(t("char_header", lang))
    tau_0  = st.number_input(t("tau0_label", lang), 1, 10, 4, 1)
    delta  = st.number_input(t("delta_label", lang), 3, 15, 7, 1)
    tau_os = st.number_input(t("tau_os_label", lang),
                              min_value=tau_0 + 1, max_value=120, value=15, step=5)

    st.divider()
    st.caption(t("sidebar_ref", lang))


# ══════════════════════════════════════════════════════════════════════════════
#  DERIVED CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

Rfw   = RFW[sort]
Rfqs  = (RFQS_G if material == "Клееная" else RFQS_S)[sort]
b_min = min(b_mm, h_mm)
v     = CHAR[(material, b_min >= 120)]
r     = h_mm / b_mm
lpf_mm = lpf_m * 1000

COND = {
    "I":   t("cond_I", lang),
    "II":  t("cond_II", lang),
    "III": t("cond_III", lang),
}

st.title(t("page_title", lang))
st.caption(t("page_caption", lang))

# ══════════════════════════════════════════════════════════════════════════════
#  3.1.1 — LOADS & FORCES
# ══════════════════════════════════════════════════════════════════════════════

qn = q_star * a_m / gamma_f
Mn = qn * L_m ** 2 / 8
Qn = qn * L_m / 2

# ══════════════════════════════════════════════════════════════════════════════
#  3.1.2 — GEOMETRIC PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════

b, h = b_mm / 1000, h_mm / 1000
W = b * h ** 2 / 6
A = b * h

# ══════════════════════════════════════════════════════════════════════════════
#  3.1.3 — CONDITION I
# ══════════════════════════════════════════════════════════════════════════════

eta_w = (Mn * 1e3) / (W * Rfw * 1e6)
p_dash_w = 0.25 / r
Z_dash_w = p_dash_w * h_mm

# ══════════════════════════════════════════════════════════════════════════════
#  3.1.4 — CONDITION II
# ══════════════════════════════════════════════════════════════════════════════

eta_A = (1.5 * Qn * 1e3) / (A * Rfqs * 1e6)
p_dash_a = 0.25 / r
Z_dash_a = p_dash_a * h_mm

# ── NOMOGRAM MANUAL INPUT ──────────────────────────────────────────────────

cola, colb = st.columns(2)

with cola:
    st.subheader(t("c1_subheader", lang))
    st.latex(
        r"\eta_w = \frac{M_n}{W \cdot R_{fw}} = "
        r"\frac{%.2f \times 10^3}{%.6f \times %.0f \times 10^6} = \mathbf{%.4f}"
        % (Mn, W, Rfw, eta_w)
    )
    st.write(t("hb_ratio", lang, hb=r))
    st.caption(t("c1_caption", lang, r=r, eta=eta_w, p=p_dash_w, z=Z_dash_w))

    use_dash_w = st.checkbox(t("dash_checkbox", lang), value=False, key="dash_w")
    if use_dash_w:
        Zcr_I = 0.25 * b_mm
        Zcr_I_note = t("dash_w_note_below", lang, z=Zcr_I)
    else:
        zh_input_I = st.number_input(
            t("zh_input_label", lang),
            min_value=0.0, max_value=0.30, value=0.020, step=0.001,
            format="%.4f", key="zh_I",
        )
        Zcr_I = zh_input_I * h_mm
        Zcr_I_note = t("zh_I_note", lang, zh=zh_input_I, z=Zcr_I)
    st.info(t("zcr_I_info", lang, z=Zcr_I))

with colb:
    st.subheader(t("c2_subheader", lang))
    st.latex(
        r"\eta_A = \frac{1.5\,Q_n}{A \cdot R_{fqs}} = "
        r"\frac{1.5 \times %.2f \times 10^3}{%.5f \times %.1f \times 10^6} = \mathbf{%.4f}"
        % (Qn, A, Rfqs, eta_A)
    )
    st.write(t("hb_ratio", lang, hb=r))
    st.caption(t("c2_caption", lang, r=r, eta=eta_A, p=p_dash_a, z=Z_dash_a))

    use_dash_a = st.checkbox(t("dash_checkbox", lang), value=False, key="dash_a")
    if use_dash_a:
        Zcr_II = 0.25 * b_mm
        Zcr_II_note = t("dash_a_note_below", lang, z=Zcr_II)
    else:
        zh_input_II = st.number_input(
            t("zh_input_label", lang),
            min_value=0.0, max_value=0.30, value=0.015, step=0.001,
            format="%.4f", key="zh_II",
        )
        Zcr_II = zh_input_II * h_mm
        Zcr_II_note = t("zh_II_note", lang, zh=zh_input_II, z=Zcr_II)
    st.info(t("zcr_II_info", lang, z=Zcr_II))

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  3.1.5 — CONDITION III
# ══════════════════════════════════════════════════════════════════════════════

Zcr1 = (tau_os - tau_0) * v
Zcr4 = 0.25 * b_mm

Kfjm = 1.0
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
    bf_mm = b_mm - k * Z_mm
    hf_mm = h_mm - n * Z_mm
    if bf_mm <= 0 or hf_mm <= 0:
        return 1e-9
    return 250 * bf_mm ** 2 / (hf_mm * lpf_mm) * Kfm_total


def eta_w_at(Z_mm: float) -> float:
    bf_mm = b_mm - k * Z_mm
    hf_mm = h_mm - n * Z_mm
    if bf_mm <= 0 or hf_mm <= 0:
        return 0.0
    return (bf_mm / b_mm) * (hf_mm / h_mm) ** 2


def sigma_fw(Z_mm: float) -> float:
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
#  RESULT
# ══════════════════════════════════════════════════════════════════════════════

cond_vals = {"I": Zcr_I, "II": Zcr_II, "III": Zcr_III}
Zcr_fin   = min(cond_vals.values())
gov       = min(cond_vals, key=cond_vals.get)

if gov == "III" and sigma_pts[0] >= Rfw:
    Pf = float(tau_os)
else:
    Pf = pf_formula(Zcr_fin, tau_0, delta, v)
fc = fire_class(Pf)

# ── Overload check ─────────────────────────────────────────────────────────
if eta_w >= 1.0:
    parts = [t("overloaded_err", lang, eta=eta_w)]
    if b_mm > h_mm:
        parts.append(t("bh_swap_hint", lang, b=int(h_mm), h=int(b_mm)))
    st.error("  \n".join(parts))
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(t("pf_label", lang), f"{Pf:.1f}")
c2.metric(t("class_label", lang), fc)
c3.metric(t("zcr_label", lang), f"{Zcr_fin:.1f}")
c4.metric(t("v_label", lang), f"{v}")
c5.metric("Rfw/Rfqs, МПа", f"{Rfw}/{Rfqs}")
st.error(t("governing_cond", lang, cond=COND[gov], z=Zcr_fin))
st.divider()

with st.expander(t("material_props", lang), expanded=False):
    ci1, ci2, ci3 = st.columns(3)
    with ci1:
        st.markdown(t("resist_header", lang))
        st.write(f"Rfw = **{Rfw} МПа** — {sort}")
        st.write(f"Rfqs = **{Rfqs} МПа** — {material_display}")
    with ci2:
        st.markdown(t("char_props", lang))
        st.write(f"υ = **{v} мм/мин**")
        st.write(f"δ = **{delta} мм**")
        st.write(f"τ₀ = **{tau_0} мин**")
    with ci3:
        st.markdown(t("stab_props", lang))
        st.write(f"k = **{k}**, n = **{n}** ({heating_label_for_display})")
        st.write(f"Kfф = **{Kff:.4f}**, Kfжм = **{Kfjm}**")
        st.write(f"lpf = **{lpf_mm:.0f} мм**")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  CROSS SECTION
# ══════════════════════════════════════════════════════════════════════════════

st.subheader(t("sec_subheader", lang))
st.caption(t("sec_caption", lang, k=k, n=n, heating=heating_label_for_display, z=Zcr_fin))

Zc = Zcr_fin
bf_mm = max(b_mm - k * Zc, 0.0)
hf_mm = max(h_mm - n * Zc, 0.0)

fig_sec = go.Figure()
fig_sec.add_shape(type="rect", x0=0, y0=0, x1=b_mm, y1=h_mm,
    line=dict(color="#999", width=1.5, dash="dot"), fillcolor="rgba(0,0,0,0)")
fig_sec.add_shape(type="rect", x0=0, y0=0, x1=b_mm, y1=h_mm,
    fillcolor="rgba(120,120,120,0.30)", line=dict(width=0))

work_color = "#A0793D"
edge_color = "#D4A017"
wall_color = "#6B4423"
delta_mm_vis = max(min(b_mm, h_mm) * 0.12, 8)

if n == 1:
    x_left, x_right = Zc, b_mm - Zc
    y_bottom, y_top = Zc, h_mm
    fig_sec.add_shape(type="rect", x0=x_left, y0=y_bottom, x1=x_right, y1=y_top,
        fillcolor=work_color, line=dict(color=edge_color, width=2))
    inner_x_left  = x_left + delta_mm_vis
    inner_x_right = x_right - delta_mm_vis
    inner_y_bottom = y_bottom + delta_mm_vis
    if inner_x_left < inner_x_right and inner_y_bottom < y_top:
        fig_sec.add_shape(type="rect", x0=inner_x_left, y0=inner_y_bottom,
            x1=inner_x_right, y1=y_top + 5,
            fillcolor="white", line=dict(color=wall_color, width=1.5))
    fig_sec.add_annotation(x=(x_left+x_right)/2, y=y_bottom+(y_top-y_bottom)*0.15,
        text=f"<b>{bf_mm:.0f}×{hf_mm:.0f} мм</b><br>{t('working_sec_u', lang)}",
        showarrow=False, font=dict(color="#7B4F00", size=12))
    fig_sec.add_annotation(x=b_mm/2, y=h_mm+16,
        text=t("protected_top", lang),
        showarrow=False, font=dict(color="#555", size=10))
else:
    x_left, x_right = Zc, b_mm - Zc
    y_bottom, y_top = Zc, h_mm - Zc
    fig_sec.add_shape(type="rect", x0=x_left, y0=y_bottom, x1=x_right, y1=y_top,
        fillcolor=work_color, line=dict(color=edge_color, width=2))
    inner_x_left  = x_left + delta_mm_vis
    inner_x_right = x_right - delta_mm_vis
    inner_y_bottom = y_bottom + delta_mm_vis
    inner_y_top    = y_top - delta_mm_vis
    if inner_x_left < inner_x_right and inner_y_bottom < inner_y_top:
        fig_sec.add_shape(type="rect", x0=inner_x_left, y0=inner_y_bottom,
            x1=inner_x_right, y1=inner_y_top,
            fillcolor="white", line=dict(color=wall_color, width=1.5))
    fig_sec.add_annotation(x=(x_left+x_right)/2, y=(y_bottom+y_top)/2,
        text=f"<b>{bf_mm:.0f}×{hf_mm:.0f} мм</b><br>{t('working_sec_box', lang)}",
        showarrow=False, font=dict(color="#7B4F00", size=12))

fig_sec.add_annotation(x=b_mm+8, y=h_mm/2, text=f"h={h_mm}мм", showarrow=False,
    font=dict(color="#333", size=11), xanchor="left")
fig_sec.add_annotation(x=b_mm/2, y=-22, text=f"b={b_mm}мм", showarrow=False,
    font=dict(color="#333", size=11))
fig_sec.add_annotation(x=2, y=max(Zc/2, 6), text=f"Z={Zc:.1f}мм", showarrow=False,
    font=dict(color="#CC3300", size=10), xanchor="left")

layout_sec = white_layout(420, title=f"k={k}, n={n}  ({heating_label_for_display})")
layout_sec["xaxis"].update(range=[-15, b_mm+75], showgrid=False, zeroline=False, showticklabels=False)
layout_sec["yaxis"].update(range=[-38, h_mm+38], showgrid=False, zeroline=False,
    showticklabels=False, scaleanchor="x", scaleratio=1)
fig_sec.update_layout(**layout_sec)
st.plotly_chart(fig_sec, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  CONDITION III CHART
# ══════════════════════════════════════════════════════════════════════════════

st.subheader(t("c3_subheader", lang))
st.caption(
    f"φfM = 250·(b−{k}·Z)² / ((h−{n}·Z)·{lpf_mm:.0f}) · {Kfm_total:.3f}   "
    f"(k={k}, n={n}, lpf={lpf_mm:.0f}мм, Kfф·Kfжм={Kfm_total:.4f})"
)

Z_line = np.linspace(Zcr1, Zcr4, 300)
sig_line = np.array([sigma_fw(z) for z in Z_line])
valid3 = (sig_line > 0) & (sig_line < 500)

fig3 = go.Figure()
fig3.add_hrect(y0=0, y1=Rfw, fillcolor="rgba(76,175,80,0.07)",
               line_width=0, annotation_text=t("safe_zone", lang),
               annotation_font_color="#2E7D32", annotation_position="top left")
if valid3.any():
    fig3.add_trace(go.Scatter(x=Z_line[valid3], y=sig_line[valid3],
        mode="lines", name="σfw(Z)", line=dict(color="#1565C0", width=3)))
fig3.add_trace(go.Scatter(x=Z_pts, y=sigma_pts,
    mode="markers+text", name=t("calc_pts", lang),
    marker=dict(color="#E65100", size=10, symbol="circle"),
    text=[f"σ={s:.1f}" for s in sigma_pts],
    textposition="top right", textfont=dict(size=11, color="#E65100")))
fig3.add_hline(y=Rfw, line=dict(color="#C62828", dash="dash", width=2),
               annotation_text=f"Rfw = {Rfw} МПа", annotation_font_color="#C62828", annotation_font_size=13)
fig3.add_vline(x=Zcr1, line=dict(color="#888", dash="dot", width=1.5),
               annotation_text=f"Zcr1={Zcr1:.1f}мм", annotation_font_color="#888")
fig3.add_vline(x=Zcr4, line=dict(color="#888", dash="dot", width=1.5),
               annotation_text=f"Zcr4={Zcr4:.0f}мм", annotation_font_color="#888")
fig3.add_vline(x=Zcr_III, line=dict(color="#E65100", width=2.5),
               annotation_text=f"Zcr={Zcr_III:.1f}мм", annotation_font_color="#E65100", annotation_font_size=13)
if Zcr1 < Zcr_III < Zcr4:
    fig3.add_trace(go.Scatter(x=[Zcr_III], y=[Rfw], mode="markers",
        marker=dict(color="#C62828", size=14, symbol="x"),
        name=f"Zcr III={Zcr_III:.1f}мм"))

y_top_g = min(float(sig_line[valid3].max() * 1.15), 60) if valid3.any() else 35
fig3.update_layout(**white_layout(380, xtitle="Z, мм", ytitle="σfw, МПа"))
fig3.update_yaxes(range=[0, y_top_g])
st.plotly_chart(fig3, use_container_width=True)

col_a, col_b, col_c, col_d = st.columns(4)
for i, (z, sig) in enumerate(zip(Z_pts, sigma_pts)):
    cols = [col_a, col_b, col_c, col_d]
    cols[i].metric(f"Z{i+1}={z:.1f}мм", f"σ={sig:.2f}МПа", delta=f"φ={phi_fM(z):.3f}")
st.info(f"**Zcr III = {Zcr_III:.1f} мм** — {III_note}")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

st.subheader(t("table_subheader", lang))
rows = []
for cid, Zcr_c in cond_vals.items():
    if cid == "III" and sigma_pts[0] >= Rfw:
        pf_c = float(tau_os)
    else:
        pf_c = pf_formula(Zcr_c, tau_0, delta, v)
    rows.append({
        t("col_cond", lang): COND[cid],
        t("col_zcr", lang):  f"{Zcr_c:.2f}",
        t("col_pf", lang):   f"{pf_c:.1f}",
        t("col_class", lang): fire_class(pf_c),
        t("col_governing", lang): t("yes", lang) if Zcr_c == Zcr_fin else "—",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP-BY-STEP SOLUTION
# ══════════════════════════════════════════════════════════════════════════════

with st.expander(t("expander_label", lang), expanded=False):

    st.markdown(t("step311", lang))
    st.latex(r"q_n = \frac{q^* \cdot a}{\gamma_f} = \frac{%.2f \cdot %.2f}{%.2f} = %.3f \text{ кН/м}" % (q_star, a_m, gamma_f, qn))
    st.latex(r"M_n = \frac{q_n L^2}{8} = \frac{%.3f \cdot %.1f^2}{8} = %.2f \text{ кН·м}" % (qn, L_m, Mn))
    st.latex(r"Q_n = \frac{q_n L}{2} = \frac{%.3f \cdot %.1f}{2} = %.2f \text{ кН}" % (qn, L_m, Qn))

    st.markdown(t("step312", lang))
    st.latex(r"W_x = \frac{b \cdot h^2}{6} = \frac{%.3f \cdot %.3f^2}{6} = %.6f \text{ м}^3" % (b, h, W))
    st.latex(r"A = b \cdot h = %.3f \cdot %.3f = %.5f \text{ м}^2" % (b, h, A))
    st.write(t("hb_ratio", lang, hb=r))

    st.markdown(t("step313", lang))
    st.latex(r"\eta_w = \frac{M_n}{W_x \cdot R_{fw}} = \frac{%.2f \times 10^3}{%.6f \times %.0f \times 10^6} = %.4f" % (Mn, W, Rfw, eta_w))
    st.write(t("rfw_table", lang, rfw=Rfw))
    st.success(f"**Z'cr = {Zcr_I:.1f} мм** — {Zcr_I_note}")

    st.markdown(t("step314", lang))
    st.latex(r"\eta_A = \frac{1.5\,Q_n}{A \cdot R_{fqs}} = \frac{1.5 \times %.2f \times 10^3}{%.5f \times %.1f \times 10^6} = %.4f" % (Qn, A, Rfqs, eta_A))
    st.write(t("rfqs_table", lang, rfqs=Rfqs))
    st.success(f"**Z''cr = {Zcr_II:.1f} мм** — {Zcr_II_note}")

    st.markdown(t("step315", lang))
    st.latex(r"Z_{cr1} = (\tau_{oc} - \tau_0)\cdot\upsilon = (%d - %d)\times %.1f = %.1f \text{ мм}" % (tau_os, tau_0, v, Zcr1))
    st.latex(r"Z_{cr4} = 0.25\,b = 0.25\times%d = %.1f \text{ мм}" % (b_mm, Zcr4))
    st.write(t("v_note", lang, v=v))
    st.write(t("arb_pts", lang, z2=Z_pts[1], z3=Z_pts[2]))
    st.markdown(f"**Kfф:** {Kff_note}")
    st.latex(r"K_{f\phi}\cdot K_{f\text{жм}} = %.4f \times %.1f = %.4f" % (Kff, Kfjm, Kfm_total))
    st.latex(r"\varphi_{fM(i)} = 250\,\frac{(b-k Z_i)^2}{(h-n Z_i)\,l_{pf}}\,K_{f\phi}K_{f\text{жм}} \qquad (k=%d,\;n=%d)" % (k, n))
    st.latex(r"\sigma_{fw(i)} = \frac{M_n}{\varphi_{fM(i)}\,W\,\eta_{w(i)}}\quad[\text{МПа}]")

    df_cols = t("df_cols", lang)
    df_pts = pd.DataFrame({
        df_cols[0]: [1, 2, 3, 4],
        df_cols[1]: [f"{z:.1f}" for z in Z_pts],
        df_cols[2]: [f"{z/h_mm:.4f}" for z in Z_pts],
        df_cols[3]: [f"{eta_w_at(z):.4f}" for z in Z_pts],
        df_cols[4]: [f"{phi_fM(z):.3f}" for z in Z_pts],
        df_cols[5]: [f"{s:.2f}" for s in sigma_pts],
    })
    st.dataframe(df_pts, use_container_width=True, hide_index=True)
    st.write(t("graph_note", lang, rfw=Rfw, note=III_note))

    st.markdown(t("step_result", lang))
    st.latex(r"Z_{cr}^{min} = \min\!\left(%.1f;\;%.1f;\;%.1f\right) = \mathbf{%.1f}\text{ мм}" % (Zcr_I, Zcr_II, Zcr_III, Zcr_fin))
    if gov == "III" and sigma_pts[0] >= Rfw:
        st.latex(r"\Pi_\phi = \tau_{oc} = \mathbf{%.0f \text{ мин}}" % tau_os)
    elif Zcr_fin > delta:
        st.latex(r"\Pi_\phi = \tau_0 + \frac{Z_{cr} - \delta}{\upsilon} = %d + \frac{%.1f - %d}{%.1f} = \mathbf{%.1f \text{ мин}}" % (tau_0, Zcr_fin, delta, v, Pf))
    else:
        st.latex(r"\Pi_\phi = Z_{cr}\cdot\frac{\tau_0}{\delta} = %.1f \cdot \frac{%d}{%d} = \mathbf{%.1f \text{ мин}}" % (Zcr_fin, tau_0, delta, Pf))
    st.success(t("conclusion", lang, fc=fc, pf=Pf, cond=COND[gov]))

st.divider()
st.caption(t("footer", lang))
