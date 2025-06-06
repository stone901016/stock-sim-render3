from flask import Flask, render_template, request, Response, stream_with_context
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from deep_translator import GoogleTranslator
import io, base64, json, os

# 字型設定（確保你有上傳 NotoSansTC-Regular.otf）
font_path = './NotoSansTC-Regular.otf'
font_prop = fm.FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()
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

    industry_en = info.get('industry', 'N/A')
    summary_en = info.get('longBusinessSummary', '')
    translator = GoogleTranslator(source='auto', target='zh-TW')
    industry = translator.translate(industry_en) if industry_en != 'N/A' else industry_en
    summary = translator.translate(summary_en[:4000]) if summary_en else '無可用公司簡介'
    summary = summary.replace('惠丘市', '新竹市')

    hist = ticker.history(period="max", auto_adjust=True)
    current_price = hist['Close'].iloc[-1]
    prices = hist['Close'].values
    returns = np.diff(prices) / prices[:-1]
    mu = returns.mean() * 252
    sigma = returns.std(ddof=1) * np.sqrt(252)
    days = HORIZON_MAP.get(horizon_key, 252)
    sims = int(sims)
    chunk = max(1, sims // 100)

    sample_n = min(2000, sims)
    sample_chunks = []
    sum_paths = np.zeros(days + 1)
    finals = []

    yield f"data: 資料下載中，請稍候\n\n"

    for i in range(0, sims, chunk):
        cnt = min(chunk, sims - i)
        rand = np.random.normal(size=(cnt, days))
        dt = 1 / 252
        inc = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand
        paths = current_price * np.exp(np.cumsum(inc, axis=1))
        paths = np.concatenate([np.full((cnt, 1), current_price), paths], axis=1)

        sum_paths += paths.sum(axis=0)
        finals.extend(paths[:, -1].tolist())

        if sample_chunks:
            already = len(np.vstack(sample_chunks))
        else:
            already = 0
        if already < sample_n:
            need = sample_n - already
            sample_chunks.append(paths[:need])

        pct = int(min((i + cnt) / sims * 100, 100))
        yield f"data: {pct}\n\n"

    # 技術指標
    delta = hist['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi14 = 100 - 100 / (1 + rs.iloc[-1])
    ema12 = hist['Close'].ewm(span=12).mean()
    ema26 = hist['Close'].ewm(span=26).mean()
    macd_val = ema12.iloc[-1] - ema26.iloc[-1]
    signal = (ema12 - ema26).ewm(span=9).mean().iloc[-1]
    macd_hist = macd_val - signal
    ma20 = hist['Close'].rolling(20).mean().iloc[-1]
    ma50 = hist['Close'].rolling(50).mean().iloc[-1]

    sample_paths = np.vstack(sample_chunks) if sample_chunks else np.empty((0, days + 1))
    mean_path = sum_paths / sims
    x = np.arange(days + 1)

    chart_title = f"{symbol} 預測未來股價 {horizon_key} ({sims:,} 次模擬)"

    fig, ax = plt.subplots(figsize=(14, 7))
    if sample_paths.size:
        ax.plot(x, sample_paths.T, lw=0.5, alpha=0.02, color='#007bff')
    ax.plot(x, mean_path, lw=3, color='#dc3545', label='平均路徑')
    ax.set_title(chart_title, fontsize=20, fontweight='bold', pad=20, fontproperties=font_prop)
    ax.set_xlabel("時間 (天)", fontsize=16, fontproperties=font_prop)
    ax.set_ylabel("價格 (元)", fontsize=16, fontproperties=font_prop)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=14, prop=font_prop)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    plot_img = base64.b64encode(buf.getvalue()).decode()

    finals = np.array(finals)
    avg_price = finals.mean()
    min_price = finals.min()
    max_price = finals.max()
    vol = finals.std()
    vol_pct = vol / avg_price * 100

    trend_text = "多頭" if ma20 > ma50 else "空頭"
    main_text = "向上" if ma20 > ma50 else "向下"
    rsi_text = "過熱" if rsi14 > 70 else ("超賣" if rsi14 < 30 else "中性")
    macd_text = "多頭動能" if macd_hist > 0 else "空頭動能"
    if ma20 > ma50 and macd_hist > 0:
        advice = "建議逢低分批買入，並於突破近期高點時加碼；如跌破下方支撐，考慮停損出場。"
    elif ma20 < ma50 and macd_hist < 0:
        advice = "建議逢高出脫，並於關鍵支撐反彈時再進場。"
    else:
        advice = "建議維持觀望，待指標趨勢更明確後再行操作。"

    commentary_html = f"""
    <div style='font-size:1rem; line-height:1.5;'>
    <h4>研究員觀點</h4>
    <h5>公司產業與業務</h5>
    <p>該公司屬於 <strong>{industry}</strong> 產業，主要業務：{summary}</p>
    <h5>模擬走勢總結</h5>
    <ul>
      <li>平均最終價格： <strong>{avg_price:.2f} 元</strong>；範圍：{min_price:.2f}-{max_price:.2f} 元；波動度：{vol:.2f} 元 ({vol_pct:.2f}%)。</li>
    </ul>
    <h5>指標解讀</h5>
    <ul>
      <li>20日MA={ma20:.2f}, 50日MA={ma50:.2f} ⇒ 趨勢偏{trend_text}，主趨勢{main_text}。</li>
      <li>RSI(14)={rsi14:.2f} ⇒ {rsi_text}，{ '注意回檔' if rsi14>70 else ('可能反彈' if rsi14<30 else '穩定') }。</li>
      <li>MACD柱狀圖={macd_hist:.4f} ⇒ {macd_text}。</li>
    </ul>
    <h5>建議</h5>
    <p>{advice}</p>
    </div>
    """

    result = {
        "title": chart_title,
        "plot_img": f"data:image/png;base64,{plot_img}",
        "hist_data": finals.tolist(),
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
    return Response(
        stream_with_context(simulate_generator(sym, hor, sims)),
        content_type="text/event-stream"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
