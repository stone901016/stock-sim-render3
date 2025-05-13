from flask import Flask, render_template, request, Response, stream_with_context
import yfinance as yf
import numpy as np
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
    # 取得公司資訊並翻譯
    ticker = yf.Ticker(symbol)
    info = ticker.info
    industry_en = info.get('industry', 'N/A')
    summary_en = info.get('longBusinessSummary', '')
    trans = GoogleTranslator(source='auto', target='zh-TW')
    industry = trans.translate(industry_en) if industry_en != 'N/A' else industry_en
    summary = trans.translate(summary_en[:4000]) if summary_en else '無可用公司簡介'
    summary = summary.replace('惠丘市', '新竹市')

    # 歷史價格與參數
    hist = ticker.history(period="max", auto_adjust=True)
    prices = hist['Close'].values
    current_price = prices[-1]
    returns = np.diff(prices) / prices[:-1]
    mu = np.mean(returns) * 252
    sigma = np.std(returns, ddof=1) * np.sqrt(252)
    days = HORIZON_MAP.get(horizon_key, 252)
    dt = 1 / 252.0

    sims = int(sims)
    chunk = max(1, sims // 100)

    # 準備圖表
    x = np.arange(days + 1)
    fig, ax = plt.subplots(figsize=(14, 7))
    sum_paths = np.zeros(days + 1)
    finals = []

    # 分批模擬並即時繪製
    for i in range(0, sims, chunk):
        cnt = min(chunk, sims - i)
        rand = np.random.normal(size=(cnt, days))
        increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand
        sims_paths = current_price * np.exp(np.cumsum(increments, axis=1))
        sims_paths = np.concatenate([np.full((cnt, 1), current_price), sims_paths], axis=1)

        # 繪製此批每條路徑
        for path in sims_paths:
            ax.plot(x, path, lw=0.5, alpha=0.015, color='#007bff')

        # 累加以計算平均路徑
        sum_paths += sims_paths.sum(axis=0)
        finals.extend(sims_paths[:, -1].tolist())

        # 回報進度
        pct = int(min((i + cnt) / sims * 100, 100))
        yield f"data: {pct}\n\n"

    # 繪製平均路徑
    mean_path = sum_paths / sims
    ax.plot(x, mean_path, lw=4, color='#dc3545', label='平均路徑')

    # 技術指標計算
    finals_arr = np.array(finals)
    avg_price = finals_arr.mean()
    min_price = finals_arr.min()
    max_price = finals_arr.max()
    vol = finals_arr.std()
    vol_pct = vol / avg_price * 100

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

    # 圖表美化
    ax.set_title(f"{symbol} 模擬走勢 ({horizon_key}, {sims} 次)", fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel("時間 (天)", fontsize=16)
    ax.set_ylabel("價格 (元)", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=14)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
    plt.close(fig)
    plot_img = base64.b64encode(buf.getvalue()).decode()

    # 自動交易建議
    if ma20 > ma50 and macd_hist > 0 and rsi14 < 70:
        advice = "趨勢偏多，建議逢低分批買入，並依多頭動能維持持有。"
    elif ma20 > ma50 and rsi14 >= 70:
        advice = "多頭趨勢明顯但 RSI 過熱，建議待整理或回測均線後分批布局。"
    elif ma20 < ma50 and macd_hist < 0:
        advice = "趨勢轉空，建議逢高分批賣出，並於支撐反彈再行進場。"
    else:
        advice = "指標混合訊號，建議先觀望，待趨勢動能更明確後再行操作。"

    # 研究員觀點 HTML
    commentary_html = f"""
    <div style="font-size:1rem; line-height:1.5;">
      <h4>公司產業與業務</h4>
      <p>該公司屬於 <strong>{industry}</strong> 產業，主要業務：{summary}</p>
      <h4>模擬走勢總結</h4>
      <ul>
        <li>預測未來股價平均約為 <strong>{avg_price:.2f} 元</strong>；範圍 {min_price:.2f}-{max_price:.2f} 元。</li>
        <li>波動度：{vol:.2f} 元 ({vol_pct:.2f}%)。</li>
      </ul>
      <h4>指標解讀</h4>
      <ul>
        <li>20 日 MA={ma20:.2f}，50 日 MA={ma50:.2f} → 趨勢偏 {'多頭' if ma20>ma50 else '空頭'}。</li>
        <li>RSI(14)={rsi14:.2f} → {'過熱易回檔' if rsi14>70 else ('超賣易反彈' if rsi14<30 else '中性穩定')}</li>
        <li>MACD 柱狀圖={macd_hist:.4f} → {'多頭動能' if macd_hist>0 else '空頭動能'}</li>
      </ul>
      <h4>建議</h4>
      <p>{advice}</p>
    </div>
    """

    result = {
        "plot_img": f"data:image/png;base64,{plot_img}",
        "hist_data": finals,
        "commentary_html": commentary_html
    }
    yield f"data: {json.dumps(result)}\n\n"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stock_stream")
def stock_stream():
    symbol = request.args.get("symbol")
    horizon = request.args.get("horizon")
    sims = request.args.get("simulations")
    return Response(
        stream_with_context(simulate_generator(symbol, horizon, sims)),
        content_type="text/event-stream"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
