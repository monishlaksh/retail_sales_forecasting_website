import warnings
warnings.filterwarnings("ignore")

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Retail Sales Forecasting")
st.subheader("XGBoost vs LSTM vs ARIMA")
st.write(
    "Upload the Kaggle Store Sales training CSV, choose a forecast horizon, "
    "and compare three time-series forecasting approaches."
)

st.info(
    "Dataset: Kaggle Store Sales – Time Series Forecasting. "
    "Upload train.csv with at least `date` and `sales` columns."
)

with st.sidebar:
    st.header("Forecast settings")
    forecast_days = st.slider("Forecast horizon (days)", 7, 90, 30, 1)
    test_fraction = st.slider("Test split (%)", 10, 40, 20, 5) / 100
    run_xgb = st.checkbox("Run XGBoost", value=True)
    run_lstm = st.checkbox("Run LSTM", value=True)
    run_arima = st.checkbox("Run ARIMA", value=True)
    lstm_epochs = st.slider("LSTM training epochs", 5, 50, 20, 5)
    arima_p = st.number_input("ARIMA p", min_value=0, max_value=10, value=5)
    arima_d = st.number_input("ARIMA d", min_value=0, max_value=2, value=1)
    arima_q = st.number_input("ARIMA q", min_value=0, max_value=10, value=0)

uploaded_file = st.file_uploader("Upload Kaggle train.csv", type=["csv"])

def load_daily_series(file):
    df = pd.read_csv(file)
    required = {"date", "sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "CSV must contain columns named `date` and `sales`. "
            f"Missing: {', '.join(sorted(missing))}"
        )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["date", "sales"])
    if df.empty:
        raise ValueError("No valid rows found in the uploaded CSV.")
    daily = df.groupby("date")["sales"].sum().sort_index().astype(float)
    idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(idx, fill_value=0.0)
    daily.index.name = "date"
    daily.name = "sales"
    return daily

def make_xgb_frame(series):
    data = pd.DataFrame({"sales": series.astype(float)})
    idx = data.index
    data["day"] = idx.day
    data["dayofweek"] = idx.dayofweek
    data["month"] = idx.month
    data["quarter"] = idx.quarter
    data["dayofyear"] = idx.dayofyear
    data["weekofyear"] = idx.isocalendar().week.astype(int).to_numpy()
    data["year"] = idx.year
    for lag in [1, 7, 14, 28]:
        data[f"lag_{lag}"] = data["sales"].shift(lag)
    for window in [7, 14, 28]:
        data[f"rolling_mean_{window}"] = (
            data["sales"].shift(1).rolling(window).mean()
        )
    return data.dropna()

def xgb_predict(model, history, dates, feature_columns):
    values = list(np.asarray(history, dtype=float))
    preds = []
    for date in dates:
        row = {
            "day": date.day,
            "dayofweek": date.dayofweek,
            "month": date.month,
            "quarter": date.quarter,
            "dayofyear": date.dayofyear,
            "weekofyear": int(date.isocalendar().week),
            "year": date.year,
        }
        for lag in [1, 7, 14, 28]:
            row[f"lag_{lag}"] = values[-lag]
        for window in [7, 14, 28]:
            row[f"rolling_mean_{window}"] = float(np.mean(values[-window:]))
        pred = float(model.predict(pd.DataFrame([row], columns=feature_columns))[0])
        pred = max(0.0, pred)
        preds.append(pred)
        values.append(pred)
    return np.asarray(preds)

def lstm_predict(train_series, steps, epochs):
    # TensorFlow is imported only if LSTM is selected.
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    tf.keras.utils.set_random_seed(42)
    time_steps = 30
    if len(train_series) <= time_steps + 10:
        raise ValueError("Not enough training days for LSTM. Use more data or a shorter test split.")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(
        np.asarray(train_series).reshape(-1, 1)
    ).astype("float32")

    X, y = [], []
    for i in range(time_steps, len(scaled)):
        X.append(scaled[i-time_steps:i, 0])
        y.append(scaled[i, 0])
    X = np.asarray(X, dtype="float32").reshape(-1, time_steps, 1)
    y = np.asarray(y, dtype="float32")

    model = Sequential([
        Input(shape=(time_steps, 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X, y,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        shuffle=False,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=0
    )

    history = list(scaled.flatten())
    pred_scaled = []
    for _ in range(steps):
        seq = np.asarray(history[-time_steps:], dtype="float32").reshape(1, time_steps, 1)
        value = float(model.predict(seq, verbose=0)[0, 0])
        pred_scaled.append(value)
        history.append(value)
    pred = scaler.inverse_transform(np.asarray(pred_scaled).reshape(-1, 1)).flatten()
    return np.maximum(pred, 0.0)

def metric_row(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    nonzero = actual != 0
    mape = (
        np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100
        if nonzero.any() else np.nan
    )
    denom = np.sum(np.abs(actual))
    wape = np.sum(np.abs(actual - predicted)) / denom * 100 if denom else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "WAPE (%)": wape}

if uploaded_file is not None:
    try:
        series = load_daily_series(uploaded_file)
        if len(series) < 100:
            st.warning("This dataset has fewer than 100 daily observations. Model quality may be limited.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Daily observations", f"{len(series):,}")
        c2.metric("Start date", series.index.min().strftime("%Y-%m-%d"))
        c3.metric("End date", series.index.max().strftime("%Y-%m-%d"))

        st.markdown("### Historical sales")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(series.index, series.values, linewidth=0.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Total sales")
        ax.set_title("Daily aggregated retail sales")
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

        if st.button("🚀 Train models and compare", type="primary", use_container_width=True):
            if not (run_xgb or run_lstm or run_arima):
                st.error("Select at least one model in the sidebar.")
                st.stop()

            split_idx = int(len(series) * (1 - test_fraction))
            train = series.iloc[:split_idx]
            test = series.iloc[split_idx:]
            if len(train) < 60 or len(test) < 1:
                st.error("Not enough data for the selected split. Upload more data or change the test split.")
                st.stop()

            predictions = {}
            progress = st.progress(0)
            status = st.empty()
            selected_count = sum([run_xgb, run_lstm, run_arima])
            completed = 0

            if run_xgb:
                status.write("Training XGBoost...")
                frame = make_xgb_frame(train)
                if frame.empty:
                    st.error("Not enough training history for XGBoost lag features.")
                    st.stop()
                X_train = frame.drop(columns=["sales"])
                y_train = frame["sales"]
                model = XGBRegressor(
                    n_estimators=400,
                    learning_rate=0.03,
                    max_depth=6,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                predictions["XGBoost"] = xgb_predict(model, train.values, test.index, X_train.columns)
                completed += 1
                progress.progress(completed / selected_count)

            if run_lstm:
                status.write("Training LSTM (this may take a few minutes)...")
                try:
                    predictions["LSTM"] = lstm_predict(train.values, len(test), lstm_epochs)
                except ImportError:
                    st.error("TensorFlow is not installed. Install it with: pip install tensorflow")
                    st.stop()
                completed += 1
                progress.progress(completed / selected_count)

            if run_arima:
                status.write("Fitting ARIMA...")
                arima_fit = ARIMA(
                    train,
                    order=(int(arima_p), int(arima_d), int(arima_q)),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit()
                predictions["ARIMA"] = np.maximum(
                    np.asarray(arima_fit.forecast(steps=len(test)), dtype=float), 0.0
                )
                completed += 1
                progress.progress(completed / selected_count)

            status.success("Model training and forecasting completed.")

            metric_data = []
            for name, pred in predictions.items():
                row = {"Model": name}
                row.update(metric_row(test.values, pred))
                metric_data.append(row)
            metrics_df = pd.DataFrame(metric_data).set_index("Model").sort_values("RMSE")

            st.markdown("### Model performance")
            st.dataframe(metrics_df.style.format("{:.3f}"), use_container_width=True)
            best_name = metrics_df["RMSE"].idxmin()
            st.success(f"Lowest RMSE on this test period: **{best_name}**")

            st.markdown("### Actual vs predicted sales")
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(test.index, test.values, label="Actual", color="black", linewidth=2)
            for name, pred in predictions.items():
                ax.plot(test.index, pred, label=name, alpha=0.85)
            ax.set_xlabel("Date")
            ax.set_ylabel("Sales")
            ax.set_title("Actual vs forecasted sales on the holdout period")
            ax.legend()
            ax.grid(alpha=0.25)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("### Error comparison")
            fig, ax = plt.subplots(figsize=(8, 4))
            metrics_df["RMSE"].plot(kind="bar", ax=ax)
            ax.set_ylabel("RMSE (lower is better)")
            ax.set_title("RMSE by model")
            ax.tick_params(axis="x", rotation=0)
            st.pyplot(fig)
            plt.close(fig)

            results = pd.DataFrame({"Actual": test.values}, index=test.index)
            for name, pred in predictions.items():
                results[f"{name}_Forecast"] = pred
            results.index.name = "Date"

            st.markdown("### Forecast output table")
            st.dataframe(results.head(30), use_container_width=True)

            st.download_button(
                "⬇️ Download forecast results (CSV)",
                data=results.to_csv().encode("utf-8"),
                file_name="retail_sales_forecast_results.csv",
                mime="text/csv"
            )
            st.download_button(
                "⬇️ Download model metrics (CSV)",
                data=metrics_df.to_csv().encode("utf-8"),
                file_name="retail_sales_model_metrics.csv",
                mime="text/csv"
            )

            st.caption(
                "Metrics are calculated on the chronological holdout period. "
                "MAPE excludes dates with zero actual sales; WAPE is included as an additional measure. "
                "The winner is data- and split-dependent, not guaranteed in advance."
            )

        st.markdown("---")
        st.markdown(
            "**PICO mapping**  \n"
            "**P:** Low accuracy and challenges in retail sales forecasting  \n"
            "**I:** XGBoost and LSTM forecasting  \n"
            "**C:** ARIMA forecasting  \n"
            "**O:** Improved forecasting accuracy, evaluated using error metrics"
        )

    except Exception as exc:
        st.error(f"Could not process the uploaded file: {exc}")
else:
    st.markdown("### How to get the dataset")
    st.markdown(
        "1. Download `train.csv` from the "
        "[Kaggle Store Sales competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).  \n"
        "2. Upload that CSV above.  \n"
        "3. Choose models and settings in the left sidebar.  \n"
        "4. Click **Train models and compare**."
    )

st.markdown("---")
st.caption("Academic prototype • Retail Sales Forecasting • XGBoost | LSTM | ARIMA")
