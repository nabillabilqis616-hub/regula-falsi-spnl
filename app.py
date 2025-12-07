# app.py
import streamlit as st
import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="SPNL - Regula Falsi", layout="centered")

st.title("Solusi SPNL — Metode Regula Falsi (False Position)")
st.write(
    "Aplikasi ini menghitung akar persamaan non-linear tunggal menggunakan metode Regula Falsi. "
    "Masukkan fungsi f(x) (contoh: `x**3 - x - 2`) dan interval [a, b]."
)

# Sidebar inputs
st.sidebar.header("Pengaturan")
default_expr = "x**3 - x - 2"
fx = st.sidebar.text_input("Masukkan f(x)", value=default_expr, help="Contoh: x**3 - x - 2")
a = st.sidebar.number_input("Lower bound (a)", value=1.0, format="%.6f")
b = st.sidebar.number_input("Upper bound (b)", value=2.0, format="%.6f")
tol = st.sidebar.number_input("Toleransi |f(c)|", value=1e-6, format="%.10f")
max_iter = st.sidebar.number_input("Max iterasi", value=50, min_value=1, step=1)

st.sidebar.markdown("---")
st.sidebar.write("Output dapat diunduh sebagai CSV setelah komputasi.")

# Helper: compile user function safely using sympy
def make_func(expr_str):
    x = sp.symbols("x")
    try:
        expr = sp.sympify(expr_str)        # parse expression safely
    except (sp.SympifyError, Exception) as e:
        raise ValueError(f"Ekspresi tidak valid: {e}")
    f_numeric = sp.lambdify(x, expr, "numpy")
    return expr, f_numeric

def regula_falsi(f, a, b, tol=1e-6, max_iter=50):
    """Implements the Regula Falsi (false position) method.
    Returns a pandas.DataFrame with iteration records."""
    fa = float(f(a))
    fb = float(f(b))

    if np.sign(fa) == np.sign(fb):
        raise ValueError("f(a) dan f(b) harus memiliki tanda berlawanan (akar harus berada di interval).")

    rows = []
    c = a
    for i in range(1, max_iter + 1):
        # false position formula
        c = (a * fb - b * fa) / (fb - fa)
        fc = float(f(c))
        rows.append({"iter": i, "a": float(a), "b": float(b), "c": float(c), "f(a)": float(fa), "f(b)": float(fb), "f(c)": float(fc)})

        if abs(fc) <= tol:
            break

        # update interval
        if np.sign(fa) * np.sign(fc) < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    df = pd.DataFrame(rows)
    return df

# Main action
compute = st.button("Hitung Regula Falsi")

if compute:
    try:
        expr, fnum = make_func(fx)
    except Exception as e:
        st.error(f"Error pada parsing fungsi: {e}")
    else:
        try:
            df = regula_falsi(fnum, a, b, tol=tol, max_iter=int(max_iter))
        except Exception as e:
            st.error(f"Error perhitungan: {e}")
        else:
            st.subheader("Ringkasan")
            root = df.iloc[-1]["c"]
            froot = df.iloc[-1]["f(c)"]
            iters = int(df.iloc[-1]["iter"])
            st.success(f"Akar hampiran: x ≈ {root:.10f}")
            st.write(f"f(x) pada akar hampiran: {froot:.3e}")
            st.write(f"Jumlah iterasi: {iters}")

            st.subheader("Tabel Iterasi")
            st.dataframe(df.style.format({
                "a": "{:.8f}", "b": "{:.8f}", "c": "{:.8f}", "f(a)": "{:.3e}", "f(b)": "{:.3e}", "f(c)": "{:.3e}"
            }))

            # Download CSV
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Unduh hasil (CSV)", data=csv, file_name="regula_falsi_results.csv", mime="text/csv")

            # Plot function and iterations
            st.subheader("Grafik fungsi & iterasi")
            x_min = min(a, b) - abs(b - a) * 0.5
            x_max = max(a, b) + abs(b - a) * 0.5
            x_vals = np.linspace(x_min, x_max, 800)
            y_vals = fnum(x_vals)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x_vals, y_vals, label=f"f(x) = {sp.pretty(expr)}")
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.set_xlim(x_min, x_max)
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.grid(True)

            # plot c points
            ax.scatter(df["c"], df["f(c)"], color="red", zorder=5, label="aproksimasi c")
            ax.legend()
            st.pyplot(fig)

            st.info("Catatan: Metode Regula Falsi cocok untuk mencari akar tunggal jika f(a) dan f(b) berbeda tanda. Untuk sistem persamaan non-linear (lebih dari 1 persamaan) diperlukan metode lain.")
else:
    st.write("Tekan tombol **Hitung Regula Falsi** setelah memasukkan f(x) dan interval [a, b].")
