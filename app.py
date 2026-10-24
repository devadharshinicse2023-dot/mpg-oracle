import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from features import clean  # noqa: F401  (needed so joblib can unpickle add_features)

st.set_page_config(page_title="MPG Oracle", page_icon="🏎️", layout="wide")

st.markdown("""
<style>
.hero{padding:1.4rem 1.8rem;border-radius:18px;color:white;
background:linear-gradient(120deg,#0f2027,#203a43,#2c5364);margin-bottom:1rem}
.hero h1{margin:0;font-size:2.2rem}.hero p{margin:.3rem 0 0;opacity:.85}
.card{padding:1rem;border-radius:14px;background:#f4f7fa;text-align:center;border:1px solid #e1e8ef}
.card h3{margin:0;color:#2c5364}.card p{margin:0;color:#667}
</style>
<div class="hero"><h1>🏎️ MPG Oracle</h1>
<p>Design a car, get its fuel efficiency instantly. Powered by machine learning.</p></div>
""", unsafe_allow_html=True)


@st.cache_resource
def load():
    return joblib.load("mpg_model.joblib")


@st.cache_data
def load_data():
    return clean(pd.read_csv("auto-mpg.csv")).dropna(subset=["mpg"])


bundle, data = load(), load_data()
pipe = bundle["pipeline"]

# ---------------- sidebar: car builder ----------------
PRESETS = {
    "Custom": None,
    "🚗 Compact city car": dict(cyl=4, disp=90, hp=70, wt=2000, acc=16.0, yr=80, org="Japan"),
    "🚙 Family sedan": dict(cyl=6, disp=200, hp=100, wt=2900, acc=15.5, yr=78, org="USA"),
    "🏁 Muscle car": dict(cyl=8, disp=350, hp=165, wt=3700, acc=11.5, yr=73, org="USA"),
    "🌿 Eco hatchback": dict(cyl=4, disp=85, hp=65, wt=1900, acc=17.0, yr=82, org="Europe"),
}
with st.sidebar:
    st.header("🔧 Build your car")
    p = PRESETS[st.selectbox("Quick presets", list(PRESETS))] or {}
    cyl = st.select_slider("Cylinders", [3, 4, 5, 6, 8], p.get("cyl", 4))
    disp = st.slider("Displacement (cu in)", 60, 460, p.get("disp", 150))
    hp = st.slider("Horsepower", 40, 250, p.get("hp", 95))
    wt = st.slider("Weight (lbs)", 1500, 5200, p.get("wt", 2800), step=10)
    acc = st.slider("Acceleration (0-60 sec)", 8.0, 25.0, p.get("acc", 15.0), step=0.1)
    yr = st.slider("Model year (19xx)", 70, 82, p.get("yr", 76))
    org = st.radio("Origin", ["USA", "Europe", "Japan"],
                   index=["USA", "Europe", "Japan"].index(p.get("org", "USA")), horizontal=True)
    brand = st.selectbox("Brand", bundle["brands"], index=bundle["brands"].index("toyota")
                         if "toyota" in bundle["brands"] else 0)
    with st.expander("⛽ Cost assumptions"):
        km_year = st.number_input("Km driven per year", 1000, 50000, 12000, step=1000)
        price = st.number_input("Fuel price (₹ / litre)", 50.0, 200.0, 105.0)

row = pd.DataFrame([dict(cylinders=cyl, displacement=disp, horsepower=hp, weight=wt,
                         acceleration=acc, model_year=yr,
                         origin={"USA": 1, "Europe": 2, "Japan": 3}[org], brand=brand)])
mpg = float(pipe.predict(row)[0])
kmpl = mpg * 0.425144

tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "🏆 Model Arena", "ℹ️ About"])

# ---------------- tab 1 ----------------
with tab1:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=mpg, number={"suffix": " MPG", "font": {"size": 44}},
            gauge={"axis": {"range": [5, 50]}, "bar": {"color": "#2c5364"},
                   "steps": [{"range": [5, 18], "color": "#ffb3b3"},
                             {"range": [18, 28], "color": "#ffe9a8"},
                             {"range": [28, 50], "color": "#b9f0c6"}]}))
        fig.update_layout(height=320, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        grade, emoji = ((" Fuel Guzzler", "🔥") if mpg < 18 else (" Average", "🚗") if mpg < 28
                        else (" Efficient", "🌿") if mpg < 36 else (" Eco Champion", "🏆"))
        st.subheader(f"{emoji}{grade}")
        litres = km_year / max(kmpl, 0.1)
        a, b = st.columns(2)
        a.markdown(f"<div class='card'><h3>{kmpl:.1f}</h3><p>km / litre</p></div>", unsafe_allow_html=True)
        b.markdown(f"<div class='card'><h3>₹{litres*price:,.0f}</h3><p>fuel / year</p></div>", unsafe_allow_html=True)
        avg = data["mpg"].mean()
        st.metric("vs dataset average", f"{mpg:.1f} MPG", f"{mpg-avg:+.1f} vs {avg:.1f}")
        st.progress(min(max((mpg - 5) / 45, 0), 1.0))

    st.markdown("#### 🔮 What-if: how does weight change your mileage?")
    ws = list(range(1500, 5201, 100))
    sweep = pd.concat([row.assign(weight=w) for w in ws])
    fig2 = go.Figure(go.Scatter(x=ws, y=pipe.predict(sweep), mode="lines", line=dict(width=4, color="#2c5364")))
    fig2.add_trace(go.Scatter(x=[wt], y=[mpg], mode="markers", marker=dict(size=14, color="red"), name="Your car"))
    fig2.update_layout(height=300, xaxis_title="Weight (lbs)", yaxis_title="Predicted MPG", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### 🚘 Most similar real cars in the dataset")
    d = data.dropna(subset=["horsepower"]).copy()
    d["dist"] = ((d.weight - wt) / 800) ** 2 + ((d.horsepower - hp) / 40) ** 2 + ((d.displacement - disp) / 80) ** 2
    st.dataframe(d.nsmallest(5, "dist")[["car_name", "model_year", "cylinders", "horsepower", "weight", "mpg"]],
                 hide_index=True, use_container_width=True)

# ---------------- tab 2 ----------------
with tab2:
    st.subheader("5-fold CV + hold-out test comparison")
    res = bundle["results"]
    st.success(f"🏆 Deployed model: **{bundle['best_name']}** (lowest RMSE)")
    st.dataframe(res.style.format({c: "{:.3f}" for c in res.columns[1:]}), hide_index=True, use_container_width=True)
    st.plotly_chart(go.Figure(go.Bar(x=res["Model"], y=res["Test R2"], marker_color="#2c5364"))
                    .update_layout(yaxis_title="Test R²", height=320), use_container_width=True)

# ---------------- tab 3 ----------------
with tab3:
    st.markdown("""
**Pipeline (all automatic, inside one scikit-learn `Pipeline`)**
1. Feature engineering: power-to-weight, displacement per cylinder, car age, brand from car name
2. Missing values: median (numeric) / most-frequent (categorical) imputation
3. Scaling: `StandardScaler` for numeric features
4. Encoding: `OneHotEncoder` for origin and brand
5. Model: best of Ridge, SVR, Random Forest, Gradient Boosting
""")
