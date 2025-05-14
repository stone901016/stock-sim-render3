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
    paths = []

    # 進度回報
    for i in range(0, sims, chunk):
        cnt = min(chunk, sims - i)
        rand = np.random.normal(size=(cnt, days))
        increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand
        sims_paths = current_price * np.exp(np.cumsum(increments, axis=1))
        sims_paths = np.concatenate([np.full((cnt, 1), current_price), sims_paths], axis=1)
        paths.append(sims_paths)
        pct = int(min((i + cnt) / sims * 100, 100))
        yield f"data: {pct}\n\n"

    all_paths = np.vstack(paths)
    finals = all_paths[:, -1]
    avg_price = finals.mean()
    min_price = finals.min()
    max_price = finals.max()
    vol = finals.std()
    vol_pct = vol / avg_price * 100

    # 技術指標
    delta = hist['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    rsi14 = 100 - 100 / (1 + rs.iloc[-1])
    ema12 = hist['Close'].ewm(span=12).mean()
    ema26 = hist['Close'].ewm(span=26).mean()
    macd_hist = (ema12 - ema26).iloc[-1] - (ema12 - ema26).ewm(span=9).mean().iloc[-1]
    ma20 = hist['Close'].rolling(20).mean().iloc[-1]
    ma50 = hist['Close'].rolling(50).mean().iloc[-1]

    # 畫走勢圖：只抽樣前 2000 條來顯示
    sample_n = min(2000, all_paths.shape[0])
    idx = np.random.choice(all_paths.shape[0], sample_n, replace=False)
    sample_paths = all_paths[idx]

    x = np.arange(days + 1)
    fig, ax = plt.subplots(figsize=(14,7))
    ax.plot(x, sample_paths.T, lw=0.5, alpha=0.03, color='#007bff')
    ax.plot(x, sample_paths.mean(axis=0), lw=3, color='#dc3545', label='平均路徑')
    ax.set_title(f"{symbol} 預測走勢 ({horizon_key}, {sims} 次模擬)", fontsize=20, pad=20)
    ax.set_xlabel("時間 (天)", fontsize=16)
    ax.set_ylabel("價格 (元)", fontsize=16)
    ax.legend(fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    plot_img = base64.b64encode(buf.getvalue()).decode()

    # 畫統計圖（直方圖）
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.hist(finals, bins=50, color='#28a745', alpha=0.7)
    ax2.set_title("最終價格分布", fontsize=16)
    ax2.set_xlabel("價格 (元)")
    ax2.set_ylabel("次數")
    ax2.grid(True, linestyle='--', alpha=0.5)
    buf2 = io.BytesIO()
    fig2.tight_layout()
    fig2.savefig(buf2, format='png')
    plt.close(fig2)
    hist_img = base64.b64encode(buf2.getvalue()).decode()

    # 建議
    if ma20>ma50 and macd_hist>0 and rsi14<70:
        advice = "趨勢偏多：建議於回檔至20日均線附近分批買入，並依動能維持持有。"
    elif ma20>ma50 and rsi14>=70:
        advice = "多頭明顯但RSI過熱，建議等待整理或回檔再布局。"
    elif ma20<ma50 and macd_hist<0:
        advice = "趨勢轉空：建議於反彈至20日均線時分批賣出或觀望。"
    else:
        advice = "訊號混合，建議先觀望、待訊號明確後再操作。"

    commentary_html = f"""
      <h4>公司產業與業務</h4>
      <p>屬於 <strong>{industry}</strong> 產業，主要業務：{summary}</p>
      <h4>模擬走勢總結</h4>
      <ul>
        <li>平均最終價格：{avg_price:.2f} 元 (範圍：{min_price:.2f}-{max_price:.2f})</li>
        <li>波動度：{vol:.2f} 元 ({vol_pct:.2f}%)</li>
      </ul>
      <h4>技術指標解讀</h4>
      <ul>
        <li>20日MA={ma20:.2f}，50日MA={ma50:.2f} → 趨勢偏{'多頭' if ma20>ma50 else '空頭'}</li>
        <li>RSI(14)={rsi14:.2f} → {'過熱易回檔' if rsi14>70 else ('超賣易反彈' if rsi14<30 else '中性穩定')}</li>
        <li>MACD柱狀圖={macd_hist:.4f} → {'多頭動能' if macd_hist>0 else '空頭動能'}</li>
      </ul>
      <h4>建議</h4>
      <p>{advice}</p>
      <h4>最終價格分布</h4>
      <img src="data:image/png;base64,{hist_img}" style="max-width:100%;">
    """

    result = {
        "plot_img": f"data:image/png;base64,{plot_img}",
        "commentary_html": commentary_html
    }
    yield f"data: {json.dumps(result)}\n\n"

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
    app.run(host="0.0.0.0", port=5000)
