import os
import pandas as pd
import streamlit as st

from computations import (
    ANALYTIC_FUNCTIONS,
    InterpolationError,
    bessel,
    build_finite_differences,
    find_closest_node_index,
    gauss_backward,
    gauss_forward,
    generate_equidistant_points,
    lagrange,
    newton_backward,
    newton_forward,
    parse_float,
    stirling,
    validate_nodes,
)
from plotting import build_interpolation_plot

st.set_page_config(page_title="Интерполяция функций", layout="wide")


def to_sup(num: int) -> str:
    """Преобразует число в надстрочные символы Юникода."""
    return str(num).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))


st.sidebar.header("Параметры данных")
input_mode = st.sidebar.radio(
    "Способ задания данных:",
    ("Аналитическая функция", "Таблица вручную", "Файл (CSV)"),
)

x_raw, y_raw = None, None
analytical_callable = None

if input_mode == "Аналитическая функция":
    func_key = st.sidebar.selectbox("Функция f(x):", list(ANALYTIC_FUNCTIONS.keys()))
    analytical_callable = ANALYTIC_FUNCTIONS[func_key][0]

    col_a, col_b = st.sidebar.columns(2)
    a_input = col_a.text_input("Левая граница (a):", value="0,0")
    b_input = col_b.text_input("Правая граница (b):", value="3,141593")
    n_pts = int(st.sidebar.number_input("Число узлов (N):", min_value=2, max_value=25, value=7, step=1))

    try:
        a = parse_float(a_input)
        b = parse_float(b_input)
    except ValueError:
        st.sidebar.error("Границы a и b должны быть корректными числами.")
        st.stop()

    if a < b:
        x_raw, y_raw = generate_equidistant_points(a, b, n_pts, analytical_callable)
    else:
        st.sidebar.error("Ошибка: граница a должна быть строго меньше b.")

elif input_mode == "Таблица вручную":
    st.sidebar.markdown("Редактирование узлов:")
    init_df = pd.DataFrame({
        "x": ["0,0", "0,2", "0,4", "0,6", "0,8", "1,0"],
        "y": ["1,0", "1,2214", "1,4918", "1,8221", "2,2255", "2,7183"],
    })
    edited_df = st.sidebar.data_editor(init_df, num_rows="dynamic", use_container_width=True)
    try:
        x_raw = [parse_float(v) for v in edited_df["x"].dropna()]
        y_raw = [parse_float(v) for v in edited_df["y"].dropna()]
    except ValueError as exc:
        st.sidebar.error(f"Ошибка ввода в таблице: {exc}")
        st.stop()

else:
    file_source = st.sidebar.radio("Источник файла:", ("Предустановленный", "Загрузить файл"))
    df_file = None
    if file_source == "Предустановленный":
        preset = st.sidebar.selectbox(
            "Файл:",
            ("test_trig.csv", "test_poly.csv", "test_runge.csv", "test_invalid.csv"),
        )
        df_file = pd.read_csv(os.path.join("data", preset))
    else:
        uploaded = st.sidebar.file_uploader("Загрузка CSV:", type=["csv", "txt"])
        if uploaded:
            df_file = pd.read_csv(uploaded)

    if df_file is not None and "x" in df_file.columns and "y" in df_file.columns:
        try:
            x_raw = [parse_float(v) for v in df_file["x"].dropna()]
            y_raw = [parse_float(v) for v in df_file["y"].dropna()]
        except ValueError as exc:
            st.sidebar.error(f"Ошибка чтения чисел из файла: {exc}")
            st.stop()

if x_raw is None or y_raw is None:
    st.stop()

try:
    x_nodes, y_nodes, is_equidistant, h = validate_nodes(x_raw, y_raw)
except InterpolationError as err:
    st.error(f"Ошибка валидации данных: {err}")
    st.stop()

min_x, max_x = x_nodes[0], x_nodes[-1]
default_x = float((min_x + max_x) / 2.0)

st.sidebar.markdown("**Аргумент x*:**")
use_manual_input = st.sidebar.checkbox("Точный ввод вручную", value=False)
if use_manual_input:
    default_str = f"{default_x:.6f}".replace(".", ",")
    x_input_str = st.sidebar.text_input("Значение x*:", value=default_str, label_visibility="collapsed")
    try:
        x_target = parse_float(x_input_str)
    except ValueError:
        st.sidebar.error("Некорректное значение аргумента x*.")
        st.stop()
else:
    step_val = float(h / 50 if is_equidistant else (max_x - min_x) / 100)
    x_target = float(
        st.sidebar.slider(
            "Значение x*:",
            min_value=float(min_x),
            max_value=float(max_x),
            value=default_x,
            step=step_val,
            label_visibility="collapsed",
        )
    )

if x_target < min_x or x_target > max_x:
    st.sidebar.warning(f"Внимание: x* выходит за интервал [{min_x:.4f}, {max_x:.4f}].")

#  Вычисления
diff_table = build_finite_differences(y_nodes) if is_equidistant else None

lagrange_key = "Лагранж"
results = {lagrange_key: lagrange(x_nodes, y_nodes, x_target)}
dense_eval_funcs = {lagrange_key: lambda xi: lagrange(x_nodes, y_nodes, xi)}

newton_key, gauss_key = None, None

if is_equidistant and diff_table is not None:
    x_mid = (min_x + max_x) / 2.0
    if x_target <= x_mid:
        newton_key = "Ньютон I"
        results[newton_key] = newton_forward(x_nodes, diff_table, h, x_target)
        dense_eval_funcs[newton_key] = lambda xi: newton_forward(x_nodes, diff_table, h, xi)
    else:
        newton_key = "Ньютон II"
        results[newton_key] = newton_backward(x_nodes, diff_table, h, x_target)
        dense_eval_funcs[newton_key] = lambda xi: newton_backward(x_nodes, diff_table, h, xi)

    center_idx = find_closest_node_index(x_nodes, x_target)
    x_center = x_nodes[center_idx]
    if x_target >= x_center:
        gauss_key = "Гаусс I"
        results[gauss_key] = gauss_forward(x_nodes, diff_table, h, x_target, center_idx)
        dense_eval_funcs[gauss_key] = lambda xi: gauss_forward(x_nodes, diff_table, h, xi, center_idx)
    else:
        gauss_key = "Гаусс II"
        results[gauss_key] = gauss_backward(x_nodes, diff_table, h, x_target, center_idx)
        dense_eval_funcs[gauss_key] = lambda xi: gauss_backward(x_nodes, diff_table, h, xi, center_idx)

    stirling_key = "Стирлинг"
    results[stirling_key] = stirling(x_nodes, diff_table, h, x_target, center_idx)
    dense_eval_funcs[stirling_key] = lambda xi: stirling(x_nodes, diff_table, h, xi, center_idx)

    bessel_key = "Бессель"
    results[bessel_key] = bessel(x_nodes, diff_table, h, x_target, center_idx)
    dense_eval_funcs[bessel_key] = lambda xi: bessel(x_nodes, diff_table, h, xi, center_idx)

# Вывод
st.title("Интерполяция функций")

st.markdown("### 1. Значение и погрешность в точке $x^*$")
y_true = analytical_callable(x_target) if analytical_callable else None

best_method = min(results.keys(), key=lambda m: abs(results[m] - y_true)) if y_true is not None else lagrange_key
kpi_cols = st.columns(3)
kpi_cols[0].metric(label=f"Лучшее значение P(x*) ({best_method})", value=f"{results[best_method]:.7f}")

if y_true is not None:
    kpi_cols[1].metric(label="Истинное значение f(x*)", value=f"{y_true:.7f}")
    kpi_cols[2].metric(label="Погрешность |P - f|", value=f"{abs(results[best_method] - y_true):.2e}")
else:
    kpi_cols[1].metric(label="Шаг сетки h", value=f"{h:.6f}" if is_equidistant else "Переменный")
    kpi_cols[2].metric(label="Число узлов N", value=len(x_nodes))

if not is_equidistant:
    st.warning("Сетка неравномерная. Вычисление формул конечных разностей заблокировано.")

st.markdown("### 2. График интерполяции")

all_curve_keys = list(results.keys())
true_func_key = "f(x)"
all_options = ([true_func_key] if analytical_callable is not None else []) + all_curve_keys

default_curves = ([true_func_key] if analytical_callable is not None else []) + [lagrange_key]
if newton_key:
    default_curves.append(newton_key)
if gauss_key:
    default_curves.append(gauss_key)

selected_curves = st.multiselect(
    "Отображаемые кривые:",
    options=all_options,
    default=default_curves,
)

n_dense = 400
grid_h = (max_x - min_x) / (n_dense - 1)
x_dense = [min_x + i * grid_h for i in range(n_dense)]

dense_curves_to_plot = {}
for m_name in results.keys():
    if m_name in selected_curves:
        dense_curves_to_plot[m_name] = [dense_eval_funcs[m_name](xi) for xi in x_dense]

fig = build_interpolation_plot(
    x_nodes=x_nodes,
    y_nodes=y_nodes,
    dense_curves=dense_curves_to_plot,
    x_dense=x_dense,
    x_target=x_target,
    target_results=results,
    true_func=analytical_callable if true_func_key in selected_curves else None,
)
st.pyplot(fig)

st.markdown("### 3. Сравнение методов интерполяции")
table_rows = []
for m_name, val in results.items():
    row = {
        "Метод": m_name,
        "P(x*)": val,
    }
    if y_true is not None:
        row["f(x*)"] = y_true
        row["|P(x*) - f(x*)|"] = abs(val - y_true)
    table_rows.append(row)

df_comparison = pd.DataFrame(table_rows)
fmt_spec = {
    "P(x*)": "{:.8f}",
    "f(x*)": "{:.8f}",
    "|P(x*) - f(x*)|": "{:.2e}",
}

_, center_col, _ = st.columns([1, 2, 1])
with center_col:
    st.dataframe(
        df_comparison.style.format({k: v for k, v in fmt_spec.items() if k in df_comparison.columns}),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("### 4. Таблица конечных разностей")
if is_equidistant and diff_table is not None:
    diff_cols = ["yᵢ"] + [f"Δ{to_sup(k)}yᵢ" if k > 1 else "Δyᵢ" for k in range(1, len(x_nodes))]
    df_diff_view = pd.DataFrame(diff_table, columns=diff_cols)
    df_diff_view.insert(0, "xᵢ", x_nodes)
    st.dataframe(
        df_diff_view.style.format(lambda v: "" if v is None else f"{v:.6f}"),
        use_container_width=True,
    )
else:
    st.info("Таблица конечных разностей не строится для неравноотстоящих узлов.")
    nodes_df = pd.DataFrame({"i": list(range(len(x_nodes))), "xᵢ": x_nodes, "yᵢ": y_nodes})
    st.dataframe(nodes_df.style.format({"xᵢ": "{:.6f}", "yᵢ": "{:.6f}"}), use_container_width=False)