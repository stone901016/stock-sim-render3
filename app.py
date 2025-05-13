from flask import Flask, render_template, request, Response, stream_with_context
import yfinance as yf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator
import io, base64, json
import random

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

MAX_PLOT_LINES = 2000  # 水塘抽樣後最多畫這麼多條路徑

def simulate_generator(symbol, horizon_key, sims):
    # --- 1. 下載、翻譯公司資訊 ---
    ticker = yf.Ticker(symbol)
    info = ticker.info
    industry_en = info.get('industry', 'N/A')
    summary_en = info.get('longBusinessSummary', '')
    zh = GoogleTranslator(source='auto', target='zh-TW')
    industry = zh.translate(industry_en) if industry_en!='N/A' else industry_en
    summary = zh.translate(summary_en[:4000]) if summary_en else '無可用公司簡介'
    summary = summary.replace('惠丘市', '新竹市')

    # --- 2. 準備歷史資料、參數 ---
    hist = ticker.history(period="max", auto_adjust=True)
    prices = hist['Close'].values
    current_price = prices[-1]
    returns = np.diff(prices)/prices[:-1]
    mu    = returns.mean() * 252
    sigma = returns.std(ddof=1) * np.sqrt(252)
    days  = HORIZON_MAP.get(horizon_key, 252)
    dt    = 1/252.0
    sims  = int(sims)
    chunk = max(1, sims//100)

    # 用於最後統計
    finals = []
    # 水塘抽樣：保留最多 MAX_PLOT_LINES 條路徑
    reservoir = []

    # --- 3. SSE 回報：模擬並抽樣 ---
    total_seen = 0
    for i in range(0, sims, chunk):
        cnt = min(chunk, sims - i)
        rand = np.random.normal(size=(cnt, days))
        increments = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*rand
        paths = current_price * np.exp(np.cumsum(increments, axis=1))
        paths = np.concatenate([np.full((cnt,1), current_price), paths], axis=1)

        # 收集 finals
        finals.extend(paths[:,-1].tolist())

        # 水塘抽樣
        for path in paths:
            total_seen += 1
            if len(reservoir) < MAX_PLOT_LINES:
                reservoir.append(path)
            else:
                # 以 MAX_PLOT_LINES/total_seen 機率替換
                idx = random.randrange(total_seen)
                if idx < MAX_PLOT_LINES:
                    reservoir[idx] = path

        # 回傳進度
        pct = int(min((i+cnt)/sims*100, 100))
        yield f"data: {pct}\n\n"

    # --- 4. 繪圖：只畫水塘抽樣後的路徑 + 平均路徑 ---
    x = np.arange(days+1)
    fig, ax = plt.subplots(figsize=(14,7))
    # 畫抽樣後的路徑
    for path in reservoir:
        ax.plot(x, path, lw=0.5, alpha=0.015, color='#007bff')
    # 平均
    finals_arr = np.array(finals)
    # 為了平均路徑，我們必須重新跑一次小量模擬來計算 sum_paths
    # 或直接從 reservoir 推估，此處簡化：用 reservoir 求平均
    mean_path = np.vstack(reservoir).mean(axis=0)
    ax.plot(x, mean_path, lw=4, color='#dc3545', label='平均路徑')

    # 技術指標
    avg_p = finals_arr.mean()
    min_p = finals_arr.min()
    max_p = finals_arr.max()
    vol   = finals_arr.std()
    vol_pct = vol/avg_p*100

    delta = hist['Close'].diff()
    gain, loss = delta.clip(lower=0), (-delta).clip(upper=0)
    rs  = gain.rolling(14).mean()/loss.rolling(14).mean()
    rsi = 100-100/(1+rs.iloc[-1])
    ema12, ema26 = hist['Close'].ewm(span=12).mean(), hist['Close'].ewm(span=26).mean()
    macd_hist = (ema12-ema26).iloc[-1] - (ema12-ema26).ewm(span=9).mean().iloc[-1]
    ma20, ma50 = hist['Close'].rolling(20).mean().iloc[-1], hist['Close'].rolling(50).mean().iloc[-1]

    # 繪圖美化
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

    # 自動建議
    if ma20>ma50 and macd_hist>0 and rsi<70:
        advice = "趨勢偏多，建議逢低分批買入，並依多頭動能維持持有。"
    elif ma20>ma50 and rsi>=70:
        advice = "多頭趨勢明顯但 RSI 過熱，建議待整理或回測均線後分批布局。"
    elif ma20<ma50 and macd_hist<0:
        advice = "趨勢轉空，建議逢高分批賣出，並於支撐反彈再行進場。"
    else:
        advice = "指標混合訊號，建議先觀望，待趨勢動能更明確後再行操作。"

    commentary = f"""
    <div style="font-size:1rem; line-height:1.5;">
      <h4>公司產業與業務</h4>
      <p>該公司屬於 <strong>{industry}</strong> 產業，主要業務：{summary}</p>
      <h4>模擬走勢總結</h4>
      <ul>
        <li>預測股價平均 {avg_p:.2f} 元，範圍 {min_p:.2f}-{max_p:.2f}。</li>
        <li>波動度：{vol:.2f} 元 ({vol_pct:.2f}%)。</li>
      </ul>
      <h4>指標解讀</h4>
      <ul>
        <li>20MA={ma20:.2f},50MA={ma50:.2f} → {'多頭' if ma20>ma50 else '空頭'}。</li>
        <li>RSI(14)={rsi:.2f} → {'過熱易回檔' if rsi>70 else ('超賣易反彈' if rsi<30 else '中性')}。</li>
        <li>MACD柱狀圖={macd_hist:.4f} → {'多頭動能' if macd_hist>0 else '空頭動能'}。</li>
      </ul>
      <h4>建議</h4><p>{advice}</p>
    </div>
    """

    result = {
      "plot_img": f"data:image/png;base64,{plot_img}",
      "hist_data": finals,
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
    sims= request.args.get("simulations")
    return Response(stream_with_context(simulate_generator(sym, hor, sims)),
                    content_type="text/event-stream")


if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)
