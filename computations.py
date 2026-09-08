import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class InterpolationError(Exception):
    pass


def parse_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).strip().replace(" ", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        raise ValueError(f"Невозможно преобразовать '{value}' в число с плавающей точкой.")


def validate_nodes(
    x: Sequence[Any], y: Sequence[Any], tol: float = 1e-5
) -> Tuple[List[float], List[float], bool, float]:
    """Проверяет корректность узлов, упорядочивает по x и определяет шаг h."""
    if len(x) != len(y):
        raise InterpolationError("Количество значений X и Y должно совпадать.")
    if len(x) < 2:
        raise InterpolationError("Для интерполяции необходимо минимум 2 точки.")

    pairs = sorted(zip((parse_float(xi) for xi in x), (parse_float(yi) for yi in y)), key=lambda p: p[0])
    x_sorted = [p[0] for p in pairs]
    y_sorted = [p[1] for p in pairs]

    diffs = [x_sorted[i + 1] - x_sorted[i] for i in range(len(x_sorted) - 1)]
    for d in diffs:
        if d < 1e-12:
            raise InterpolationError("Узлы содержат совпадающие координаты X.")

    step = diffs[0]
    is_equidistant = all(abs(d - step) < tol for d in diffs)
    return x_sorted, y_sorted, is_equidistant, step


def find_closest_node_index(x_nodes: Sequence[float], x: float) -> int:
    """Определяет индекс узла, ближайшего к целевому аргументу x."""
    return min(range(len(x_nodes)), key=lambda i: abs(x_nodes[i] - x))


def build_finite_differences(y: Sequence[float]) -> List[List[Optional[float]]]:
    """Формирует треугольную матрицу конечных разностей."""
    n = len(y)
    table: List[List[Optional[float]]] = [[None] * n for _ in range(n)]

    for i in range(n):
        table[i][0] = float(y[i])

    for j in range(1, n):
        for i in range(n - j):
            prev_down = table[i + 1][j - 1]
            prev_curr = table[i][j - 1]
            if prev_down is not None and prev_curr is not None:
                table[i][j] = prev_down - prev_curr
    return table


def lagrange(x_nodes: Sequence[float], y_nodes: Sequence[float], x: float) -> float:
    n = len(x_nodes)
    total = 0.0
    for i in range(n):
        term = float(y_nodes[i])
        for j in range(n):
            if i != j:
                term *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        total += term
    return total


def newton_forward(
    x_nodes: Sequence[float],
    diff_table: List[List[Optional[float]]],
    h: float,
    x: float,
) -> float:
    n = len(x_nodes)
    q = (x - x_nodes[0]) / h
    val = diff_table[0][0]
    if val is None:
        return 0.0

    result = val
    q_term = 1.0

    for k in range(1, n):
        q_term *= (q - (k - 1))
        diff_val = diff_table[0][k]
        if diff_val is None:
            break
        result += (q_term / math.factorial(k)) * diff_val
    return result


def newton_backward(
    x_nodes: Sequence[float],
    diff_table: List[List[Optional[float]]],
    h: float,
    x: float,
) -> float:
    n = len(x_nodes)
    last = n - 1
    q = (x - x_nodes[last]) / h
    val = diff_table[last][0]
    if val is None:
        return 0.0

    result = val
    q_term = 1.0

    for k in range(1, n):
        q_term *= (q + (k - 1))
        diff_val = diff_table[last - k][k]
        if diff_val is None:
            break
        result += (q_term / math.factorial(k)) * diff_val
    return result


def gauss_forward(
    x_nodes: Sequence[float],
    diff_table: List[List[Optional[float]]],
    h: float,
    x: float,
    center_idx: int,
) -> float:
    n = len(x_nodes)
    m = center_idx
    q = (x - x_nodes[m]) / h

    val = diff_table[m][0]
    if val is None:
        return 0.0

    result = val
    q_term = 1.0

    for k in range(1, n):
        if k % 2 == 1:
            q_term *= (q + (k // 2)) if k > 1 else q
        else:
            q_term *= (q - (k // 2))

        row = m - (k // 2)
        if row < 0 or (row + k) >= n:
            break

        diff_val = diff_table[row][k]
        if diff_val is None:
            break

        result += (q_term / math.factorial(k)) * diff_val
    return result


def gauss_backward(
    x_nodes: Sequence[float],
    diff_table: List[List[Optional[float]]],
    h: float,
    x: float,
    center_idx: int,
) -> float:
    n = len(x_nodes)
    m = center_idx
    q = (x - x_nodes[m]) / h

    val = diff_table[m][0]
    if val is None:
        return 0.0

    result = val
    q_term = 1.0

    for k in range(1, n):
        if k % 2 == 1:
            q_term *= (q - (k // 2)) if k > 1 else q
        else:
            q_term *= (q + (k // 2))

        row = m - ((k + 1) // 2)
        if row < 0 or (row + k) >= n:
            break

        diff_val = diff_table[row][k]
        if diff_val is None:
            break

        result += (q_term / math.factorial(k)) * diff_val
    return result


def stirling(
    x_nodes: Sequence[float],
    diff_table: List[List[Optional[float]]],
    h: float,
    x: float,
    center_idx: int,
) -> float:
    n = len(x_nodes)
    m = center_idx
    q = (x - x_nodes[m]) / h

    val = diff_table[m][0]
    if val is None:
        return 0.0
    result = val

    p = 1
    prod_sq = 1.0

    while True:
        k_odd = 2 * p - 1
        r_odd_1 = m - (p - 1)
        r_odd_2 = m - p

        if r_odd_2 < 0 or (r_odd_1 + k_odd) >= n:
            break

        d_odd_1 = diff_table[r_odd_1][k_odd]
        d_odd_2 = diff_table[r_odd_2][k_odd]
        if d_odd_1 is None or d_odd_2 is None:
            break

        factor_odd = (q * prod_sq) / math.factorial(k_odd)
        result += factor_odd * (0.5 * (d_odd_1 + d_odd_2))

        k_even = 2 * p
        r_even = m - p

        if r_even < 0 or (r_even + k_even) >= n:
            break

        d_even = diff_table[r_even][k_even]
        if d_even is None:
            break

        factor_even = (q * q * prod_sq) / math.factorial(k_even)
        result += factor_even * d_even

        prod_sq *= (q * q - p * p)
        p += 1

    return result


def bessel(
    x_nodes: Sequence[float],
    diff_table: List[List[Optional[float]]],
    h: float,
    x: float,
    center_idx: int,
) -> float:
    n = len(x_nodes)
    if n < 2:
        return 0.0

    m = max(0, min(n - 2, center_idx))
    q = (x - x_nodes[m]) / h

    y_m = diff_table[m][0]
    y_m1 = diff_table[m + 1][0]
    if y_m is None or y_m1 is None:
        return 0.0

    result = 0.5 * (y_m + y_m1)
    prod_even = 1.0

    p = 1
    while True:
        k_odd = 2 * p - 1
        r_odd = m - (p - 1)

        if r_odd < 0 or (r_odd + k_odd) >= n:
            break

        d_odd = diff_table[r_odd][k_odd]
        if d_odd is None:
            break

        factor_odd = (q - 0.5) * prod_even / math.factorial(k_odd)
        result += factor_odd * d_odd

        k_even = 2 * p
        r_even_1 = m - (p - 1)
        r_even_2 = m - p

        if r_even_2 < 0 or (r_even_1 + k_even) >= n:
            break

        d_even_1 = diff_table[r_even_1][k_even]
        d_even_2 = diff_table[r_even_2][k_even]
        if d_even_1 is None or d_even_2 is None:
            break

        prod_even *= (q + (p - 1)) * (q - p)
        factor_even = prod_even / math.factorial(k_even)
        result += factor_even * (0.5 * (d_even_1 + d_even_2))

        p += 1

    return result


def generate_equidistant_points(
    start: float, stop: float, count: int, func: Callable[[float], float]
) -> Tuple[List[float], List[float]]:
    if count < 2:
        raise InterpolationError("Количество точек должно быть не менее 2.")
    step = (stop - start) / (count - 1)
    x = [start + i * step for i in range(count)]
    y = [func(xi) for xi in x]
    return x, y


ANALYTIC_FUNCTIONS: Dict[str, Tuple[Callable[[float], float], str]] = {
    "y = sin(x)": (math.sin, "sin(x)"),
    "y = cos(x)": (math.cos, "cos(x)"),
    "y = x^3 - 3x^2 + 2x + 1": (
        lambda x: x**3 - 3.0 * (x**2) + 2.0 * x + 1.0,
        "x^3 - 3x^2 + 2x + 1",
    ),
    "y = 1 / (1 + 25x^2) [Рунге]": (
        lambda x: 1.0 / (1.0 + 25.0 * (x**2)),
        "1 / (1 + 25x^2)",
    ),
    "y = exp(x)": (math.exp, "exp(x)"),
}