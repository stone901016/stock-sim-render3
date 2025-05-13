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
    # 取歷史、計算參數
    ticker = yf.Ticker(symbol)
    info = ticker.info
    hist = ticker.history(period="max", auto_adjust=True)
    current_price = hist['Close'].iloc[-1]
    prices = hist['Close'].values
    returns = np.diff(prices) / prices[:-1]
    mu    = np.mean(returns) * 252
    sigma = np.std(returns, ddof=1) * np.sqrt(252)
    days  = HORIZON_MAP.get(horizon_key, 252)
    dt = 1/252
    sims = int(sims)

    # 每 1% 回報一次進度
    chunk = max(1, sims // 100)
    paths = []
    for i in range(0, sims, chunk):
        cnt = min(chunk, sims - i)
        rand = np.random.normal(size=(cnt, days))
        increments = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*rand
        sims_paths = current_price * np.exp(np.cumsum(increments, axis=1))
        sims_paths = np.concatenate([np.full((cnt,1), current_price), sims_paths], axis=1)
        paths.append(sims_paths)
        pct = int(min((i+cnt)/sims*100, 100))
        yield f"data: {pct}\n\n"

    all_paths = np.vstack(paths)
    finals = all_paths[:, -1]
    avg_price = np.mean(finals)
    min_price = np.min(finals)
    max_price = np.max(finals)
    vol = np.std(finals)
    vol_pct = vol/avg_price*100

    # 技術指標
    delta = hist['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    rsi14 = 100 - 100/(1+rs.iloc[-1])
    ema12 = hist['Close'].ewm(span=12).mean()
    ema26 = hist['Close'].ewm(span=26).mean()
    macd_hist = (ema12.iloc[-1]-ema26.iloc[-1]) - (ema12-ema26).ewm(span=9).mean().iloc[-1]
    ma20 = hist['Close'].rolling(20).mean().iloc[-1]
    ma50 = hist['Close'].rolling(50).mean().iloc[-1]

    # 隨機抽樣 2000 條路徑畫圖
    n_plot = min(all_paths.shape[0], 2000)
    idx = np.random.choice(all_paths.shape[0], n_plot, replace=False)
    plot_paths = all_paths[idx]

    # 繪圖
    x = np.arange(days+1)
    fig, ax = plt.subplots(figsize=(14,7))
    ax.plot(x, plot_paths.T, lw=0.5, alpha=0.2, color='#007bff')
    ax.plot(x, np.mean(all_paths,axis=0), lw=2, color='#dc3545', label='平均路徑')
    ax.set_title(f"{symbol} 預測未來股價 {horizon_key} ({sims} 次模擬)",
                 fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel("時間 (天)", fontsize=16)
    ax.set_ylabel("價格 (元)", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=14)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
    plt.close(fig)
    plot_img = base64.b64encode(buf.getvalue()).decode()

    # 研究員觀點
    commentary = (
        f"<div style='font-size:1rem; line-height:1.5;'>"
        f"<h4>模擬走勢總結</h4>"
        f"<ul>"
        f"<li>預測平均：<strong>{avg_price:.2f} 元</strong>，範圍 {min_price:.2f}-{max_price:.2f} 元，波動度 {vol:.2f} ({vol_pct:.2f}%)。</li>"
        f"</ul>"
        f"<h4>指標解讀</h4>"
        f"<ul>"
        f"<li>20MA={ma20:.2f}, 50MA={ma50:.2f} ⇒ {'多頭' if ma20>ma50 else '空頭'}。</li>"
        f"<li>RSI(14)={rsi14:.2f} ⇒ {'過熱回檔' if rsi14>70 else ('超賣反彈' if rsi14<30 else '中性') }。</li>"
        f"<li>MACD柱狀={macd_hist:.4f} ⇒ {'多頭動能' if macd_hist>0 else '空頭動能'}。</li>"
        f"</ul>"
        f"<h4>建議</h4>"
        f"<p>{'建議逢低分批買入，並於突破時加碼；跌破支撐則停損。' if ma20>ma50 and macd_hist>0 else '建議逢高減碼或觀望。'}</p>"
        f"</div>"
    )

    result = {
      "plot_img": f"data:image/png;base64,{plot_img}",
      "hist_data": finals.tolist(),
      "commentary_html": commentary
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
    app.run(host='0.0.0.0', port=5000)
