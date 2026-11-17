from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# --- Load mô hình CatBoost theo nhóm ---
MODEL_DIR = Path(__file__).resolve().parents[1] / "Train" / "models"
MODEL_PATHS = {
    "Low": MODEL_DIR / "CatBoost_Low.pkl",
    "Medium": MODEL_DIR / "CatBoost_Mid.pkl",
    "High": MODEL_DIR / "CatBoost_High.pkl"
}
MODELS = {grp: joblib.load(path) for grp, path in MODEL_PATHS.items()}

NUM_FEATS = [
    "Price", "Age", "LogPrice", "PricePerAge", "IsOld",
    "LogNumberOfRatings", "RatingSupportScore",
    "WineryFreq", "RegionFreq", "CountryFreq", "VarietyFreq",
    "PriceAgeInteraction", "LogLogPrice",
    "Country_Score", "Region_Score", "Winery_Score", "Variety_Score"
]


def preprocess(df, varieties_list=None):
    # --- Impute Age và loại bỏ NV / thiếu dữ liệu ---
    df["IsNV"] = df["Year"] == "N.V."
    df = df[df["Year"] != "N.V."].copy()
    df["Year"] = df["Year"].astype(int)
    df["Age"] = 2025 - df["Year"]
    df = df.dropna(subset=["Rating","Price","Country","Region","Winery","WineStyle","Age"])

    # --- Extract Variety ---
    if varieties_list is None:
        varieties_list = ["Cabernet","Pinot","Merlot","Shiraz","Zinfandel"]
    def extract_variety(nm):
        for v in varieties_list:
            if isinstance(nm, str) and v.lower() in nm.lower():
                return v
        return "Unknown"
    df["Variety"] = df["Name"].apply(extract_variety)

    # --- SmoothedRating & ReliabilityGroup ---
    global_mean = df["Rating"].mean()
    df["SmoothedRating"] = (
        df["Rating"]*df["NumberOfRatings"] + global_mean*5
    ) / (df["NumberOfRatings"] + 5)
    df["ReliabilityGroup"] = df["NumberOfRatings"].apply(
        lambda n: "Low" if n<35 else ("Medium" if n<838 else "High")
    )

    # --- Basic numeric features ---
    df["LogPrice"]            = np.log1p(df["Price"])
    df["PricePerAge"]         = df["Price"] / (df["Age"]+1)
    df["IsOld"]               = (df["Age"]>10).astype(int)
    df["LogNumberOfRatings"]  = np.log1p(df["NumberOfRatings"])
    df["RatingSupportScore"]  = df["NumberOfRatings"]/(df["Age"]+1)

    # --- Frequency features ---
    df["WineryFreq"]  = df["Winery"].map(df["Winery"].value_counts()/len(df))
    df["RegionFreq"]  = df["Region"].map(df["Region"].value_counts()/len(df))
    df["CountryFreq"] = df["Country"].map(df["Country"].value_counts()/len(df))
    df["VarietyFreq"] = df["Variety"].map(df["Variety"].value_counts()/len(df))

    # --- Interaction & log-log ---
    df["PriceAgeInteraction"] = df["PricePerAge"] * df["LogPrice"]
    df["LogLogPrice"]         = np.log1p(df["LogPrice"])

    # --- Country / Region / Winery / Variety level scores ---
    # 1. Country
    c_agg = df.groupby("Country")["SmoothedRating"].mean().reset_index()
    c_agg.columns = ["Country","Country_MeanRating"]
    c_agg["Country_Rank"]  = c_agg["Country_MeanRating"].rank(method="first",ascending=False)
    max_r = c_agg["Country_Rank"].max()
    c_agg["Country_Score"] = (max_r - c_agg["Country_Rank"])/(max_r-1) if max_r>1 else 1.0
    df = df.merge(c_agg, on="Country", how="left")

    # 2. Region
    r_agg = df.groupby("Region")["SmoothedRating"].mean().reset_index()
    r_agg.columns = ["Region","Region_MeanRating"]
    r_agg["Region_Rank"]  = r_agg["Region_MeanRating"].rank(method="first",ascending=False)
    max_r = r_agg["Region_Rank"].max()
    r_agg["Region_Score"] = (max_r - r_agg["Region_Rank"])/(max_r-1) if max_r>1 else 1.0
    df = df.merge(r_agg, on="Region", how="left")

    # 3. Winery
    w_agg = df.groupby("Winery")["SmoothedRating"].mean().reset_index()
    w_agg.columns = ["Winery","Winery_MeanRating"]
    w_agg["Winery_Rank"]  = w_agg["Winery_MeanRating"].rank(method="first",ascending=False)
    max_r = w_agg["Winery_Rank"].max()
    w_agg["Winery_Score"] = (max_r - w_agg["Winery_Rank"])/(max_r-1) if max_r>1 else 1.0
    df = df.merge(w_agg, on="Winery", how="left")

    # 4. Variety
    v_agg = df.groupby("Variety")["SmoothedRating"].mean().reset_index()
    v_agg.columns = ["Variety","Variety_MeanRating"]
    v_agg["Variety_Rank"]  = v_agg["Variety_MeanRating"].rank(method="first",ascending=False)
    max_r = v_agg["Variety_Rank"].max()
    v_agg["Variety_Score"] = (max_r - v_agg["Variety_Rank"])/(max_r-1) if max_r>1 else 1.0
    df = df.merge(v_agg, on="Variety", how="left")

    # --- Fill NaN in score/rank ---
    for col in ["Country_Score","Region_Score","Winery_Score","Variety_Score",
                "Country_MeanRating","Region_MeanRating","Winery_MeanRating","Variety_MeanRating",
                "Country_Rank","Region_Rank","Winery_Rank","Variety_Rank"]:
        df[col] = df[col].fillna(df[col].median() if "Rank" in col else df[col].mean())

    # --- Clean numeric features ---
    num_cols = [
        "Price","Age","LogPrice","PricePerAge","IsOld",
        "LogNumberOfRatings","RatingSupportScore",
        "WineryFreq","RegionFreq","CountryFreq","VarietyFreq",
        "PriceAgeInteraction","LogLogPrice",
        "Country_Score","Region_Score","Winery_Score","Variety_Score",
        "RatingSupportScore"
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col],errors="coerce").fillna(0)

    # --- Ensure categorical are str ---
    for col in ["Country","Region","Winery","WineStyle","Variety"]:
        df[col] = df[col].astype(str)

    df.replace([np.inf,-np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

def predict_grouped(df):
    FEATURES = [
      "Country","Region","Winery","Price","WineStyle","Variety",
      "Age","PricePerAge","IsOld","LogPrice",
      "WineryFreq","RegionFreq","CountryFreq","VarietyFreq",
      "PriceAgeInteraction","LogLogPrice",
      "Country_Score","Region_Score","Winery_Score","Variety_Score",
      "RatingSupportScore","LogNumberOfRatings"
    ]
    df["PredictedRating"] = np.nan

    for grp in df["ReliabilityGroup"].unique():
        model = MODELS.get(grp)
        if not model:
            continue

        mask = df["ReliabilityGroup"] == grp
        X = df.loc[mask, FEATURES].copy()

        # ép cat_features
        cat_idxs = model.get_params().get("cat_features", [])
        for idx in cat_idxs:
            col = X.columns[idx]
            X[col] = X[col].astype(str)

        df.loc[mask, "PredictedRating"] = model.predict(X)

    return df



def run_app():
    st.title("🍷 Ứng dụng Dự đoán Rating Rượu Vang")

    f = st.file_uploader("📂 Tải file CSV theo cấu trúc Red.csv", type="csv")
    if not f:
        return

    # đọc CSV với fallback encoding
    try:
        df = pd.read_csv(f, encoding="utf-8")
    except:
        df = pd.read_csv(f, encoding="ISO-8859-1")

    df = preprocess(df)
    df = predict_grouped(df)
    # fallback dùng SmoothedRating nếu predict thiếu
    df["PredictedRating"] = df["PredictedRating"].fillna(df["SmoothedRating"])
    # tính delta tiềm năng
    df["Delta"] = df["PredictedRating"] - df["SmoothedRating"]

    tab1, tab2, tab3 = st.tabs([
        "🏆 Top tiềm năng tốt",
        "⚙️ Điều chỉnh giá & tuổi",
        "⚠️ Top tiềm năng kém"
    ])

    # 1) Top tiềm năng tốt
    with tab1:
        n_top = st.number_input("Số lượng muốn xem", 1, len(df), 10, key="n_top_good")
        df_top = df.sort_values("Delta", ascending=False).head(n_top).copy()
        df_top = df_top.rename(columns={
            "SmoothedRating": "Rating cũ",
            "PredictedRating": "Rating mới"
        })
        # round 1 chữ số
        df_top["Rating cũ"] = df_top["Rating cũ"].round(1)
        df_top["Rating mới"] = df_top["Rating mới"].round(1)
        df_top["Delta"]     = df_top["Delta"].round(1)
        st.dataframe(df_top[["Name","Country","Price","Rating cũ","Rating mới","Delta"]])

 # --- Tab2: điều chỉnh giá & tuổi ---
    with tab2:
        # chọn vùng & rượu
        region = st.selectbox("Chọn vùng", df["Region"].unique())
        df_reg = df[df["Region"] == region]
        wine = st.selectbox("Chọn rượu trong vùng", df_reg["Name"])
        row  = df_reg[df_reg["Name"] == wine].iloc[0]

        st.write(f"**Rating cũ:** {row['PredictedRating']:.1f}  |  **Smoothed:** {row['SmoothedRating']:.1f}")
        col_price_old, col_age_old = st.columns(2)
        with col_price_old:
            st.metric("💲 Giá cũ", f"{row['Price']:.2f}")
        with col_age_old:
            st.metric("⏳ Tuổi cũ", f"{int(row['Age'])} năm")
        price_new = st.number_input("Giá mới ($)", min_value=0.0, value=float(row["Price"]), key="price_new")
        age_new   = st.number_input("Tuổi mới (năm)", min_value=0, max_value=500, value=int(row["Age"]), key="age_new")

        # build toàn bộ features giống predict_grouped
        Xn = pd.DataFrame([{
            "Country":       row["Country"],
            "Region":        row["Region"],
            "Winery":        row["Winery"],
            "WineStyle":     row["WineStyle"],
            "Variety":       row["Variety"],
            "Price":         price_new,
            "Age":           age_new,
            "LogPrice":      np.log1p(price_new),
            "PricePerAge":   price_new/(age_new+1),
            "IsOld":         int(age_new>10),
            "LogNumberOfRatings": np.log1p(row["NumberOfRatings"]),
            "RatingSupportScore": row["NumberOfRatings"]/(age_new+1),
            "WineryFreq":    row["WineryFreq"],
            "RegionFreq":    row["RegionFreq"],
            "CountryFreq":   row["CountryFreq"],
            "VarietyFreq":   row["VarietyFreq"],
            "PriceAgeInteraction": (price_new/(age_new+1))*np.log1p(price_new),
            "LogLogPrice":   np.log1p(np.log1p(price_new)),
            "Country_Score": row["Country_Score"],
            "Region_Score":  row["Region_Score"],
            "Winery_Score":  row["Winery_Score"],
            "Variety_Score": row["Variety_Score"],
            "SmoothedRating": row["SmoothedRating"],        # nếu model dùng
            "LogLogPrice":    np.log1p(np.log1p(price_new))
        }])

        # giữ đúng thứ tự feature như predict_grouped
        FEATURES = [
        "Country","Region","Winery","Price","WineStyle","Variety",
        "Age","PricePerAge","IsOld","LogPrice",
        "WineryFreq","RegionFreq","CountryFreq","VarietyFreq",
        "PriceAgeInteraction","LogLogPrice",
        "Country_Score","Region_Score","Winery_Score","Variety_Score",
        "RatingSupportScore","LogNumberOfRatings"
        ]
        Xn = Xn[FEATURES]

        # TỰ ĐỘNG ép tất cả cat_features (model.params_["cat_features"]) về string
        model = MODELS[row["ReliabilityGroup"]]
        cat_idxs = model.get_params().get("cat_features", [])
        for idx in cat_idxs:
            col = Xn.columns[idx]
            Xn[col] = Xn[col].astype(str)

        # predict
        pred_new = model.predict(Xn)[0]

        # hiển thị
        col1, col2 = st.columns(2)
        with col1:
            # 1 chữ số
            st.metric("⭐ Rating cũ", f"{row['PredictedRating']:.1f}")
        with col2:
            st.metric("📈 Rating mới",  f"{pred_new:.1f}")


    # 3) Top tiềm năng kém
    with tab3:
        n_bad = st.number_input("Số lượng muốn xem", 1, len(df), 10, key="n_bad_bad")
        df_bad = df.sort_values("Delta", ascending=True).head(n_bad).copy()
        df_bad = df_bad.rename(columns={
            "SmoothedRating": "Rating cũ",
            "PredictedRating": "Rating mới"
        })
        df_bad["Rating cũ"] = df_bad["Rating cũ"].round(1)
        df_bad["Rating mới"] = df_bad["Rating mới"].round(1)
        df_bad["Delta"]     = df_bad["Delta"].round(1)
        st.dataframe(df_bad[["Name","Country","Price","Rating cũ","Rating mới","Delta"]])

if __name__ == "__main__":
    run_app()
