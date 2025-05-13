# app.py
from flask import Flask, render_template, request, Response, stream_with_context
import yfinance as yf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, json, random

app = Flask(__name__)

HORIZON_MAP = {
    "1D": 1, "1W": 5, "1M": 21, "3M": 63,
    "6M": 126, "1Y": 252, "2Y": 504,
    "5Y": 1260, "10Y": 2520,
}

def simulate_generator(symbol, horizon_key, sims, plot_max=2000):
    sims = int(sims)
    days = HORIZON_MAP.get(horizon_key, 252)
    dt = 1/252

    # 取得歷史資料，計算 S0, μ, σ
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="max", auto_adjust=True)
    prices = hist['Close'].values
    S0 = prices[-1]
    returns = np.diff(prices)/prices[:-1]
    mu    = returns.mean()*252
    sigma = returns.std(ddof=1)*np.sqrt(252)

    # 提前抽樣哪些 index 要畫走勢圖
    sample_idx = set(random.sample(range(sims), min(plot_max, sims)))

    # 先準備 finals array
    finals = np.empty(sims, dtype=float)
    # reservoir for sampled paths
    reservoir = []

    # 分批模擬並回報進度
    chunk = max(1, sims // 100)
    for i in range(0, sims, chunk):
        cnt = min(chunk, sims - i)
        # 模擬 cnt 條路徑
        Z = np.random.normal(size=(cnt, days))
        inc = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
        sims_paths = S0 * np.exp(np.cumsum(inc, axis=1))
        sims_paths = np.hstack([np.full((cnt,1), S0), sims_paths])

        # 記錄 finals
        finals[i:i+cnt] = sims_paths[:,-1]

        # 抽樣保留
        for k in range(cnt):
            gi = i + k
            if gi in sample_idx:
                reservoir.append(sims_paths[k])

        # 回報進度
        pct = int(min((i+cnt)/sims*100, 100))
        yield f"data: {pct}\n\n"

    # 完成後繪圖：畫出 reservoir 中的 sample 路徑與其平均
    sample_paths = np.vstack(reservoir) if reservoir else np.empty((0, days+1))
    x = np.arange(days+1)
    fig, ax = plt.subplots(figsize=(12,6))
    for row in sample_paths:
        ax.plot(x, row, lw=0.5, alpha=0.02, color='#007bff')
    if len(sample_paths)>0:
        ax.plot(x, sample_paths.mean(axis=0), lw=2.5, color='#dc3545', label='平均路線')
    ax.set_title(f"{symbol} 預測未來股價 {horizon_key} ({sims:,} 次模擬)", fontsize=18, fontweight='bold')
    ax.set_xlabel("時間 (天)")
    ax.set_ylabel("價格 (元)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    plot_img = base64.b64encode(buf.getvalue()).decode()

    # 最後一次回傳 JSON 結果
    avg_price = float(finals.mean())
    vol       = float(finals.std())
    vol_pct   = vol/avg_price*100
    result = {
        "plot_img": f"data:image/png;base64,{plot_img}",
        "hist_data": finals.tolist(),
        "summary": {
            "avg": round(avg_price,2),
            "std": round(vol,2),
            "vol_pct": round(vol_pct,2),
            "count": sims
        }
    }
    yield f"data: {json.dumps(result)}\n\n"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stock_stream")
def stock_stream():
    sym = request.args.get("symbol")
    hor = request.args.get("horizon")
    sims= request.args.get("simulations")
    return Response(
        stream_with_context(simulate_generator(sym, hor, sims)),
        content_type="text/event-stream"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
