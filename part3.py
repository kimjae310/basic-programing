import tkinter as tk
from tkinter import messagebox, ttk
import random
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime
import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import math

#💾 상태 저장 파일 불러오기 또는 초기화
state_file = "portfolio.json"
if os.path.exists(state_file):
    with open(state_file, "r") as f:
        state = json.load(f)
else:
    state = {
        "cash": 1000000.0,
        "positions": {},
        "prediction": [],
        "dates": [],
        "ticker": "",
        "current_day": 0
    }

#🔮 예측만 수행하는 함수 (기존 매수/매도용 예측)
def predict_prices(ticker):
    start_date = (datetime.datetime.today() - datetime.timedelta(days=180)).strftime('%Y-%m-%d')
    df = fdr.DataReader(ticker, start_date)[["Close"]]
    df.rename(columns={"Close": "종가"}, inplace=True)
    df.index.name = "일자"

    if len(df) < 30:
        messagebox.showerror("오류", "데이터가 부족합니다.")
        return None

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    window_size = 20
    forecast_size = 5
    X, y = [], []
    for i in range(len(scaled) - window_size - forecast_size + 1):
        X.append(scaled[i:i+window_size])
        y.append(scaled[i+window_size:i+window_size+forecast_size].flatten())
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(y.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=300, batch_size=16, verbose=0)

    recent_data = df.values[-window_size:]
    input_scaled = scaler.transform(recent_data).reshape(1, window_size, 1)
    pred_scaled = model.predict(input_scaled)
    prediction = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten().tolist()

    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=5, freq='B')
    state["prediction"] = prediction
    state["dates"] = [d.strftime('%Y-%m-%d') for d in future_dates]
    state["ticker"] = ticker
    state["current_day"] = 0
    save_state()

    result_text = f"[{ticker}] 예측된 향후 5일 종가:\n"
    for i, p in enumerate(state["prediction"], 1):
        result_text += f"Day {i}: ₩{p:,.2f}\n"
    prediction_result.config(text=result_text)
    return prediction

#📊 예측과 실제 비교

def predict_and_evaluate(ticker):
    today = datetime.datetime.today()
    train_end = today - datetime.timedelta(days=7)
    test_start = train_end + datetime.timedelta(days=1)
    start_date = train_end - datetime.timedelta(days=180)

    df = fdr.DataReader(ticker, start_date.strftime('%Y-%m-%d'))[["Close"]]
    df.rename(columns={"Close": "종가"}, inplace=True)
    df.index.name = "일자"

    df_train = df[df.index <= train_end]
    df_test = df[(df.index >= test_start) & (df.index <= today)]

    if len(df_train) < 30 or len(df_test) < 5:
        messagebox.showerror("오류", "데이터가 부족합니다.")
        return

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_train)
    X, y = [], []
    window_size = 20
    forecast_size = 5
    for i in range(len(scaled) - window_size - forecast_size + 1):
        X.append(scaled[i:i+window_size])
        y.append(scaled[i+window_size:i+window_size+forecast_size].flatten())
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(y.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=300, batch_size=16, verbose=0)

    recent_data = df_train.values[-window_size:]
    input_scaled = scaler.transform(recent_data).reshape(1, window_size, 1)
    pred_scaled = model.predict(input_scaled)
    prediction = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten().tolist()
    actual = df_test['종가'].values[:len(prediction)]

    mae = mean_absolute_error(actual, prediction)
    rmse = math.sqrt(mean_squared_error(actual, prediction))

    result_text = f"[{ticker}] 예측 vs 실제 (최근 5일):\n"
    for i in range(len(actual)):
        result_text += f"Day {i+1}: 예측={prediction[i]:,.2f}, 실제={actual[i]:,.2f}\n"
    result_text += f"\n📊 MAE: {mae:.2f}  |  RMSE: {rmse:.2f}"
    prediction_result.config(text=result_text)

    fig.clear()
    ax = fig.add_subplot(111)
    future_dates = df_test.index[:len(prediction)]
    ax.plot(future_dates, actual, label="실제", marker='o', color='blue')
    ax.plot(future_dates, prediction, label="예측", marker='o', color='orange')
    ax.set_title(f"{ticker} 예측 vs 실제")
    ax.set_ylim(min(min(actual), min(prediction)) * 0.95, max(max(actual), max(prediction)) * 1.05)
    ax.legend()
    canvas.draw()

#💰 포트폴리오 관련

def save_state():
    with open(state_file, "w") as f:
        json.dump(state, f)

def get_price(index):
    if index < len(state["prediction"]):
        return round(state["prediction"][index], 2)
    return 0.0

def refresh_ui():
    cash_label.config(text=f"💰 현금: ${state['cash']:.2f}")
    day_label.config(text=f"📆 현재 예측일: Day {state['current_day']+1 if state['prediction'] else 0}")
    portfolio_table.delete(*portfolio_table.get_children())
    for tkr, pos in state["positions"].items():
        for i in range(len(pos["qty"])):
            price = pos["price"][i]
            qty = pos["qty"][i]
            current_price = get_price(state["current_day"])
            profit = (current_price - price) * qty
            portfolio_table.insert("", "end", values=(tkr, f"Day{state['current_day']+1}", qty, f"${price:.2f}", f"${current_price:.2f}", f"${profit:.2f}"))
    update_chart()

def buy():
    tkr = ticker_entry.get().strip()
    index = state["current_day"]
    try:
        qty = int(qty_entry.get())
    except:
        messagebox.showerror("오류", "수량을 정확히 입력하세요")
        return
    price = get_price(index)
    total = price * qty
    if state["cash"] >= total:
        if tkr not in state["positions"]:
            state["positions"][tkr] = {"qty": [], "price": []}
        state["positions"][tkr]["qty"].append(qty)
        state["positions"][tkr]["price"].append(price)
        state["cash"] -= total
        save_state()
        refresh_ui()
    else:
        messagebox.showerror("오류", "잔액 부족")

def sell():
    tkr = ticker_entry.get().strip()
    index = state["current_day"]
    try:
        qty = int(qty_entry.get())
    except:
        messagebox.showerror("오류", "수량을 정확히 입력하세요")
        return
    if tkr not in state["positions"]:
        messagebox.showerror("오류", "보유 중인 종목이 없습니다")
        return
    found = False
    for i in range(len(state["positions"][tkr]["qty"])):
        if state["positions"][tkr]["qty"][i] >= qty:
            found = True
            price = get_price(index)
            state["positions"][tkr]["qty"][i] -= qty
            state["cash"] += price * qty
            if state["positions"][tkr]["qty"][i] == 0:
                state["positions"][tkr]["qty"].pop(i)
                state["positions"][tkr]["price"].pop(i)
            if not state["positions"][tkr]["qty"]:
                del state["positions"][tkr]
            break
    if not found:
        messagebox.showerror("오류", "해당 수량 매도 불가")
    save_state()
    refresh_ui()

def next_day():
    if state["current_day"] < len(state["prediction"]) - 1:
        state["current_day"] += 1
        save_state()
        refresh_ui()
    else:
        messagebox.showinfo("알림", "예측된 마지막 날입니다.")

def update_chart():
    fig.clear()
    ax = fig.add_subplot(111)
    if state["prediction"]:
        ax.plot(state["dates"], state["prediction"], marker='o', label=state["ticker"], color='orange')
        ax.axvline(state["dates"][state["current_day"]], color='red', linestyle='--', label='오늘')
        ax.set_title(f"{state['ticker']} 향후 5일 예측")
        ax.set_ylim(min(state["prediction"])*0.95, max(state["prediction"])*1.05)
        ax.legend()
    canvas.draw()

#🖥️ GUI 구성
root = tk.Tk()
root.title("📈 LSTM 기반 주가 예측 + 모의투자 + 정확도 평가")
root.geometry("1000x800")

predict_frame = tk.Frame(root)
predict_frame.pack(pady=10)
tk.Label(predict_frame, text="종목 코드 입력 (예: 005930)").pack()
ticker_entry = tk.Entry(predict_frame)
ticker_entry.pack()
tk.Button(predict_frame, text="🔮 예측 실행", command=lambda: [predict_prices(ticker_entry.get()), refresh_ui()]).pack(pady=5)
tk.Button(predict_frame, text="📊 정확도 평가", command=lambda: predict_and_evaluate(ticker_entry.get())).pack(pady=3)

prediction_result = tk.Label(predict_frame, text="", justify='left', font=("Courier", 12))
prediction_result.pack()

trade_frame = tk.Frame(root)
trade_frame.pack(pady=10)
day_label = tk.Label(trade_frame, text="")
day_label.pack()
tk.Label(trade_frame, text="수량 입력").pack()
qty_entry = tk.Entry(trade_frame)
qty_entry.pack()
tk.Button(trade_frame, text="📥 매수", command=buy).pack(pady=3)
tk.Button(trade_frame, text="📤 매도", command=sell).pack(pady=3)
tk.Button(trade_frame, text="➡️ 다음날", command=next_day).pack(pady=3)
cash_label = tk.Label(trade_frame, text="")
cash_label.pack(pady=5)

cols = ("종목", "예측일", "수량", "매수가", "예측가", "손익")
portfolio_table = ttk.Treeview(root, columns=cols, show="headings")
for col in cols:
    portfolio_table.heading(col, text=col)
    portfolio_table.column(col, width=120)
portfolio_table.pack(fill="x", pady=10)

chart_frame = tk.Frame(root)
chart_frame.pack(fill="both", expand=True)
fig = plt.Figure(figsize=(6, 3), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=chart_frame)
canvas.get_tk_widget().pack(fill="both", expand=True)

refresh_ui()
root.mainloop()
