import streamlit as st
import yfinance as yf
import pandas as pd
import time

# 포트폴리오 초기화
def init_portfolio():
    return {'cash': 1000000.0, 'positions': {}, 'history': [], 'bought_at': None}

def buy_stock(portfolio, ticker, price, qty):
    try:
        price = float(price)
        qty = int(qty)
        cost = float(price * qty)
        cash = portfolio.get('cash', 0)
        if hasattr(cash, 'item'):
            cash = cash.item()
        if hasattr(cost, 'item'):
            cost = cost.item()

        if float(cash) >= float(cost):
            portfolio['cash'] = float(cash - cost)
            portfolio['positions'][ticker] = {'qty': qty, 'avg_price': price}
            portfolio['history'].append({'ticker': ticker, 'price': price, 'qty': qty, 'type': 'BUY'})
            portfolio['bought_at'] = time.time()
            return True
        else:
            return False
    except Exception as e:
        st.error(f"[매수 실패 - 내부오류] {str(e)}")
        return False

def sell_stock(portfolio, ticker, price):
    try:
        price = float(price)
        if ticker in portfolio['positions']:
            qty = int(portfolio['positions'][ticker]['qty'])
            proceeds = price * qty
            portfolio['cash'] += proceeds
            portfolio['history'].append({'ticker': ticker, 'price': price, 'qty': qty, 'type': 'SELL'})
            del portfolio['positions'][ticker]
            return True
        return False
    except Exception as e:
        st.error(f"[매도 실패] {e}")
        return False

def evaluate_position(portfolio, current_price):
    results = []
    current_price = float(current_price)
    for ticker, pos in portfolio['positions'].items():
        qty = int(pos['qty'])
        avg = float(pos['avg_price'])
        value = current_price * qty
        cost = avg * qty
        profit = value - cost
        rate = (profit / cost) * 100
        results.append({
            'ticker': ticker,
            'qty': qty,
            'avg_price': avg,
            'current_price': current_price,
            'eval_value': value,
            'profit': profit,
            'rate': rate
        })
    return results

# Streamlit 시작
st.set_page_config(page_title="📊 실시간 모의투자 시뮬레이션")
st.title("📊 예측 결과 기반 실시간 모의투자 시뮬레이션")

# 세션 상태 관리
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = init_portfolio()
if 'start_index' not in st.session_state:
    st.session_state['start_index'] = 60

portfolio = st.session_state['portfolio']

# 종목 선택 및 데이터 로딩
selected = st.selectbox("종목 선택", ['AAPL', 'MSFT', 'TSLA'])
df = yf.download(selected, period="30d", interval="1h")
df = df[['Close']].dropna().reset_index(drop=True)

st.line_chart(df['Close'])

# 매수
st.subheader("💰 매수")
qty = st.number_input("수량 입력", min_value=1, value=1, step=1)
buy_price = float(df['Close'].iloc[st.session_state['start_index']])

if st.button("매수 실행"):
    if buy_stock(portfolio, selected, buy_price, qty):
        st.success(f"✅ {selected} {qty}주 매수 완료 (단가: ${buy_price:.2f})")
    else:
        st.error("❌ 잔액 부족 또는 매수 실패")

# 실시간 시뮬레이션 시작
time_placeholder = st.empty()
chart_placeholder = st.empty()
status_placeholder = st.empty()
eval_placeholder = st.empty()
sell_placeholder = st.empty()

if portfolio['positions']:
    for i in range(st.session_state['start_index'], len(df)):
        current_price = float(df['Close'].iloc[i])
        time_placeholder.markdown(f"⏱️ 시뮬레이션 시세: `${current_price:.2f}`")
        chart_placeholder.line_chart(df['Close'].iloc[st.session_state['start_index']:i+1])

        evaluation = evaluate_position(portfolio, current_price)
        for e in evaluation:
            eval_placeholder.markdown(
                f"📦 보유: {e['ticker']} | 수익률: {e['rate']:.2f}% | 평가손익: ${e['profit']:.2f}"
            )
            if e['rate'] >= 5:
                status_placeholder.success("📤 수익률 5% 도달! 매도 추천")
            elif e['rate'] <= -3:
                status_placeholder.warning("📉 손실 -3% 이상, 손절 고려")

        # 매도 버튼 제공
        if selected in portfolio['positions']:
            if sell_placeholder.button(f"💸 {selected} 전량 매도하기 (현재가 ${current_price:.2f})"):
                if sell_stock(portfolio, selected, current_price):
                    st.success(f"💰 {selected} 매도 완료! 현재 자산: ${portfolio['cash']:.2f}")
                    break

        time.sleep(0.5)
else:
    st.info("먼저 매수하고 시뮬레이션을 시작하세요!")
