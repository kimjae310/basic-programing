import streamlit as st
import json
import pandas as pd

# 포트폴리오 초기화
def init_portfolio():
    return {'cash': 1000000, 'positions': {}, 'history': []}

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
        portfolio['history'].append({'ticker': ticker, 'price': price, 'qty': qty, 'type': 'BUY'})
        return True
    return False

# Streamlit 시작
st.title(\"📊 예측 결과 기반 모의투자 시스템\")

# 포트폴리오 세션 상태
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = init_portfolio()
portfolio = st.session_state['portfolio']

# 예측 결과 로딩
try:
    with open('prediction_result.json', 'r') as f:
        predictions = json.load(f)
    pred = predictions[0]
    ticker = pred['ticker']
    last_price = pred['last_price']
    pred_price = pred['predicted_price']
    diff = pred_price - last_price
    trend = \"📈 매수 추천\" if diff > 0 else \"📉 보류\"

    st.subheader(f\"🔍 {ticker} 예측 결과\")    
    st.metric(\"현재가\", f\"${last_price:.2f}\")
    st.metric(\"예측 종가\", f\"${pred_price:.2f}\", delta=f\"{diff:.2f}\")
    st.write(f\"추천 의견: {trend}\")

    st.subheader(\"💰 모의투자\")    
    qty = st.number_input(\"매수 수량\", min_value=1, step=1)
    if st.button(\"매수 실행\"):
        if buy_stock(portfolio, ticker, last_price, qty):
            st.success(f\"{ticker} {qty}주 매수 완료!\")
        else:
            st.error(\"잔액 부족!\")

    st.subheader(\"📦 보유 종목\")
    if portfolio['positions']:
        df_pos = pd.DataFrame.from_dict(portfolio['positions'], orient='index')
        df_pos['현재가'] = last_price
        df_pos['평가금액'] = df_pos['현재가'] * df_pos['qty']
        df_pos['수익률'] = ((df_pos['현재가'] - df_pos['avg_price']) / df_pos['avg_price']) * 100
        st.dataframe(df_pos.round(2))
    else:
        st.info(\"현재 보유 종목 없음.\")

except FileNotFoundError:
    st.error(\"❌ 예측 결과 파일(prediction_result.json)이 없습니다. Colab이나 학습 스크립트를 먼저 실행해 주세요.\")
