import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st
import datetime

# -----------------------------
# 1. 데이터 로딩 함수
# -----------------------------
def load_data(ticker='AAPL', period='6mo'):
    df = yf.download(ticker, period=period)
    df = df[['Close']].dropna()
    return df

# -----------------------------
# 2. 전처리 함수 (LSTM 학습용)
# -----------------------------
def prepare_data(df, lookback=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y), scaler

# -----------------------------
# 3. LSTM 모델 훈련
# -----------------------------
def train_lstm(X, y):
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    return model

# -----------------------------
# 4. 예측 함수
# -----------------------------
def predict_price(model, X_last, scaler):
    pred_scaled = model.predict(X_last.reshape(1, -1, 1))
    pred = scaler.inverse_transform(pred_scaled)
    return float(pred[0][0])

# -----------------------------
# 5. 모의투자 기록 관리
# -----------------------------
def init_portfolio():
    return {
        'cash': 1000000,
        'positions': {},
        'history': []
    }

def buy_stock(portfolio, ticker, price, qty):
    cost = price * qty
    if portfolio['cash'] >= cost:
        portfolio['cash'] -= cost
        if ticker in portfolio['positions']:
            pos = portfolio['positions'][ticker]
            total_qty = pos['qty'] + qty
            avg_price = (pos['avg_price'] * pos['qty'] + price * qty) / total_qty
            pos['qty'] = total_qty
            pos['avg_price'] = avg_price
        else:
            portfolio['positions'][ticker] = {'qty': qty, 'avg_price': price}
        portfolio['history'].append({
            'date': str(datetime.date.today()),
            'ticker': ticker,
            'price': price,
            'qty': qty,
            'type': 'BUY'
        })
        return True
    return False

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="📈 LSTM 모의투자 시스템")
st.title("📊 LSTM 기반 종목 예측 & 모의투자")

stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
selected = st.selectbox("종목 선택", stocks)

# 포트폴리오 세션 관리
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = init_portfolio()
portfolio = st.session_state['portfolio']

# 데이터 로딩 및 모델 학습
df = load_data(selected)
X, y, scaler = prepare_data(df)
model = train_lstm(X, y)
X_last = X[-1]
pred_price = predict_price(model, X_last, scaler)
last_price = df['Close'].iloc[-1]
diff = pred_price - last_price
trend = "📈 매수 추천" if diff > 0 else "📉 보류"

st.subheader(f"🔎 예측 결과: {selected}")
st.metric("현재가", f"${last_price:.2f}")
st.metric("예측 종가", f"${pred_price:.2f}", delta=f"{diff:.2f}")
st.write(f"추천 의견: {trend}")

# 차트
st.line_chart(df['Close'])

# 모의 투자 기능
st.subheader("💰 모의투자")
buy_qty = st.number_input("매수 수량", min_value=1, step=1)
if st.button("매수 실행"):
    success = buy_stock(portfolio, selected, last_price, buy_qty)
    if success:
        st.success(f"✅ {selected} 주식 {buy_qty}주 매수 완료!")
    else:
        st.error("❌ 잔액 부족! 매수 실패")

# 포트폴리오 보기
st.subheader("📦 보유 종목")
if portfolio['positions']:
    df_pos = pd.DataFrame.from_dict(portfolio['positions'], orient='index')
    df_pos['현재가'] = last_price
    df_pos['평가금액'] = df_pos['현재가'] * df_pos['qty']
    df_pos['수익률'] = ((df_pos['현재가'] - df_pos['avg_price']) / df_pos['avg_price']) * 100
    st.dataframe(df_pos.round(2))
else:
    st.info("현재 보유 중인 종목이 없습니다.")

st.caption("💡 예측 기반 투자 결과는 실제 투자 결과와 다를 수 있습니다.")
