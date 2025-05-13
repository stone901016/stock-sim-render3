from flask import Flask, render_template, request, Response, stream_with_context
import yfinance as yf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, json

app = Flask(__name__)

# 模擬時間映射
HORIZON_MAP = {
    "1D": 1, "1W": 5, "1M": 21, "3M": 63,
    "6M": 126, "1Y": 252, "2Y": 504,
    "5Y": 1260, "10Y": 2520,
}

def simulate_generator(symbol, horizon_key, sims, plot_max=2000):
    sims = int(sims)
    days = HORIZON_MAP.get(horizon_key, 252)
    dt = 1/252

    # 取得歷史資料、計算 μ, σ
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="max", auto_adjust=True)
    prices = hist['Close'].values
    mu = np.mean(np.diff(prices)/prices[:-1]) * 252
    sigma = np.std(np.diff(prices)/prices[:-1], ddof=1) * np.sqrt(252)
    S0 = prices[-1]

    # 分批模擬以回報進度
    chunk = max(1, sims // 100)
    all_paths = []
    for i in range(0, sims, chunk):
        cnt = min(chunk, sims - i)
        # GBM 增量
        Z = np.random.normal(size=(cnt, days))
        inc = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
        sim_paths = S0 * np.exp(np.cumsum(inc, axis=1))
        sim_paths = np.hstack([np.full((cnt,1), S0), sim_paths])
        all_paths.append(sim_paths)
        pct = int(min((i+cnt)/sims*100, 100))
        yield f"data: {pct}\n\n"

    all_paths = np.vstack(all_paths)  # shape=(sims, days+1)
    finals = all_paths[:,-1]
    avg_price = finals.mean()
    vol = finals.std()
    vol_pct = vol/avg_price*100

    # 走勢圖：抽樣 plot_max 條路線＋平均路線
    sample_idx = np.linspace(0, sims-1, min(plot_max, sims), dtype=int)
    sample_paths = all_paths[sample_idx]
    x = np.arange(days+1)
    fig, ax = plt.subplots(figsize=(12,6))
    for row in sample_paths:
        ax.plot(x, row, lw=0.5, alpha=0.02, color='#007bff')
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

    # 直方圖資料（全部 finals）
    # 前端用 Chart.js 做 bar chart
    result = {
        "plot_img": f"data:image/png;base64,{plot_img}",
        "hist_data": finals.tolist(),
        "summary": {
            "avg": round(float(avg_price),2),
            "std": round(float(vol),2),
            "vol_pct": round(float(vol_pct),2),
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
    sims = request.args.get("simulations")
    return Response(
        stream_with_context(simulate_generator(sym, hor, sims)),
        content_type="text/event-stream"
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
