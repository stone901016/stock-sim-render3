from flask import Flask, render_template, request, Response, stream_with_context
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator
import io, base64, json

# 中文字體設定
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

app = Flask(__name__)

HORIZON_MAP = {
    "1D": 1, "1W": 5, "1M": 21, "3M": 63,
    "6M": 126, "1Y": 252, "2Y": 504,
    "5Y": 1260, "10Y": 2520,
}

def simulate_generator(symbol, horizon_key, sims):
    ticker = yf.Ticker(symbol)
    info = ticker.info

    # 翻譯公司產業與業務簡介
    industry_en = info.get('industry', 'N/A')
    summary_en = info.get('longBusinessSummary', '')
    translator = GoogleTranslator(source='auto', target='zh-TW')
    industry = translator.translate(industry_en) if industry_en != 'N/A' else industry_en
    summary = translator.translate(summary_en[:4000]) if summary_en else '無可用公司簡介'
    summary = summary.replace('惠丘市', '新竹市')

    # 歷史資料
    hist = ticker.history(period="max", auto_adjust=True)
    current_price = hist['Close'].iloc[-1]
    prices = hist['Close'].values
    returns = np.diff(prices) / prices[:-1]
    mu = np.mean(returns) * 252
    sigma = np.std(returns, ddof=1) * np.sqrt(252)
    days = HORIZON_MAP.get(horizon_key, 252)
    dt = 1/252
    sims = int(sims)
    chunk = max(1, sims // 100)
    finals = np.empty((0,))

    # 分批模擬並回傳進度
    for i in range(0, sims, chunk):
        cnt = min(chunk, sims - i)
        rand = np.random.normal(size=(cnt, days))
        increments = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*rand
        sims_paths = current_price * np.exp(np.cumsum(increments, axis=1))
        sims_paths = np.hstack([np.full((cnt,1), current_price), sims_paths])
        finals = np.concatenate([finals, sims_paths[:,-1]])
        pct = int((i+cnt)/sims*100)
        yield f"data: {pct}\n\n"

    avg_price = finals.mean()
    min_price = finals.min()
    max_price = finals.max()
    vol = finals.std()
    vol_pct = vol / avg_price * 100

    # 技術指標
    delta = hist['Close'].diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean(); avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss; rsi14 = 100 - 100/(1+rs.iloc[-1])
    ema12 = hist['Close'].ewm(span=12).mean(); ema26 = hist['Close'].ewm(span=26).mean()
    macd_hist = (ema12.iloc[-1]-ema26.iloc[-1]) - ((ema12-ema26).ewm(span=9).mean().iloc[-1])
    ma20 = hist['Close'].rolling(20).mean().iloc[-1]
    ma50 = hist['Close'].rolling(50).mean().iloc[-1]

    # 繪圖：改用 finals 而非 all_paths，減少記憶體
    fig, ax = plt.subplots(figsize=(14,7))
    # 畫模擬平均路徑
    ax.plot(range(days+1),
            np.linspace(current_price, avg_price, days+1),
            lw=4, color='#dc3545', label='平均路徑')
    ax.set_title(f"{symbol} 預測未來股價 {horizon_key} ({sims} 次模擬)", fontsize=20, fontweight='bold')
    ax.set_xlabel("時間 (天)", fontsize=16)
    ax.set_ylabel("價格 (元)", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=14)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
    plt.close(fig)
    plot_img = base64.b64encode(buf.getvalue()).decode()

    # 建議文字
    if ma20>ma50 and macd_hist>0 and rsi14<70:
        advice = "建議逢低分批買入，並於突破近期高點時加碼；如跌破下方支撐，考慮停損出場。"
    elif ma20>ma50 and rsi14>=70:
        advice = "多頭明顯但 RSI 過熱，建議等待回檔或整理後再布局。"
    elif ma20<ma50 and macd_hist<0:
        advice = "逢反彈至20日均線時分批賣出或觀望以控風險。"
    else:
        advice = "訊號混合，建議觀望並待指標明確後再操作。"

    # HTML 組合
    commentary_html = f"""
<div style="font-size:1rem; line-height:1.5;">
  <h4>公司產業與業務</h4>
  <p>該公司屬於 <strong>{industry}</strong> 產業，主要業務：{summary}</p>
  <h4>模擬走勢總結</h4>
  <ul>
    <li>預測未來股價平均約 <strong>{avg_price:.2f} 元</strong>，範圍 {min_price:.2f}–{max_price:.2f} 元，波動度 {vol_pct:.2f}%。</li>
  </ul>
  <h4>指標解讀</h4>
  <ul>
    <li>20日MA={ma20:.2f}, 50日MA={ma50:.2f} ⇒ 趨勢{"多頭" if ma20>ma50 else "空頭"}。</li>
    <li>RSI(14)={rsi14:.2f} ⇒ {"過熱(>70)" if rsi14>70 else ("超賣(<30)" if rsi14<30 else "中性")}。</li>
    <li>MACD 柱狀圖={macd_hist:.4f} ⇒ {"多頭動能" if macd_hist>0 else "空頭動能"}。</li>
  </ul>
  <h4>建議</h4>
  <ul><li>{advice}</li></ul>
</div>
"""

    payload = {
        "plot_img": f"data:image/png;base64,{plot_img}",
        "hist_data": finals.tolist(),
        "commentary_html": commentary_html
    }
    yield f"data: {json.dumps(payload)}\n\n"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stock_stream")
def stock_stream():
    sym = request.args.get("symbol")
    hor = request.args.get("horizon")
    sims = request.args.get("simulations")
    return Response(stream_with_context(simulate_generator(sym, hor, sims)),
                    content_type="text/event-stream")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
