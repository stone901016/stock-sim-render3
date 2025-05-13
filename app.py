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
    summary_en = info.get('longBusinessSummary', '無可用公司簡介')
    translator = GoogleTranslator(source='auto', target='zh-TW')
    industry = translator.translate(industry_en) if industry_en != 'N/A' else industry_en
    # deep-translator一次最多翻譯5000字，業務簡介可能過長，可截斷
    summary = translator.translate(summary_en[:4000]) if summary_en else '無可用公司簡介'
    summary = summary.replace('惠丘市', '新竹市')  # 修正誤翻譯城市名稱

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

    for i in range(0, sims, chunk):
        cnt = min(chunk, sims - i)
        rand = np.random.normal(size=(cnt, days))
        increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand
        sims_paths = current_price * np.exp(np.cumsum(increments, axis=1))
        sims_paths = np.concatenate([np.full((cnt, 1), current_price), sims_paths], axis=1)
        paths.append(sims_paths)
        percent = int(min((i + cnt) / sims * 100, 100))
        yield f"data: {percent}\n\n"

    all_paths = np.vstack(paths)
    finals = all_paths[:, -1]
    avg_price = np.mean(finals)
    min_price = np.min(finals)
    max_price = np.max(finals)
    vol = np.std(finals)
    vol_pct = vol / avg_price * 100

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
    signal_line = (ema12 - ema26).ewm(span=9).mean().iloc[-1]
    macd_hist = macd_val - signal_line
    ma20 = hist['Close'].rolling(20).mean().iloc[-1]
    ma50 = hist['Close'].rolling(50).mean().iloc[-1]

    # 繪圖
    x = np.arange(days + 1)
    fig, ax = plt.subplots(figsize=(14,7))
    ax.plot(x, all_paths.T, lw=0.5, alpha=0.015, color='#007bff')
    ax.plot(x, np.mean(all_paths, axis=0), lw=4, color='#dc3545', label='平均路徑')
    ax.set_title(f"{symbol} 預測未來股價 {horizon_key} ({sims}次模擬)", fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel("時間 (天)", fontsize=16)
    ax.set_ylabel("價格 (元)", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=14)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
    plt.close(fig)
    plot_img = base64.b64encode(buf.getvalue()).decode()

    
    # 自動產生具體操作建議
    if ma20 > ma50 and macd_hist > 0 and rsi14 < 70:
        suggestion = "趨勢偏多: 建議於股價回檔至20日均線附近分批買入，並依當前多頭動能保持持有。"
    elif ma20 > ma50 and rsi14 >= 70:
        suggestion = "雖多頭趨勢明顯，但 RSI 過熱，建議等待股價整理或回檔至均線位置再分批布局。"
    elif ma20 < ma50 and macd_hist < 0:
        suggestion = "趨勢轉空: 建議於股價反彈至20日均線時分批賣出或觀望以控制風險。"
    else:
        suggestion = "指標呈現混合訊號，建議先觀望並待趨勢及動能訊號更為明確後再行操作。"

    commentary_html = (
        f"<div style='font-size:1rem; line-height:1.5;'>"
        f"<h4>公司產業與業務</h4>"
        f"<p>該公司屬於 <strong>{industry}</strong> 產業，主要業務：{summary}</p>"
        f"<h4>模擬走勢總結</h4>"
        f"<ul>"
        f"<li>預測未來股價平均約為 <strong>{avg_price:.2f} 元</strong>。</li>"
        f"<li>目前股價：{current_price:.2f} 元；預測範圍：{min_price:.2f}-{max_price:.2f} 元。</li>"
        f"<li>波動度：{vol:.2f} 元 ({vol_pct:.2f}%)。</li>"
        f"</ul>"
        f"<h4>指標解讀</h4>"
        f"<ul>"
        f"<li>20日MA={ma20:.2f}, 50日MA={ma50:.2f} => 趨勢偏{'多頭' if ma20>ma50 else '空頭'}。</li>"
        f"<li>RSI(14)={rsi14:.2f} => {'過熱(>70)易回檔' if rsi14>70 else ('超賣(<30)易反彈' if rsi14<30 else '中性穩定')}</li>"
        f"<li>MACD柱狀圖={macd_hist:.4f} => {'正值，多頭動能' if macd_hist>0 else '負值，空頭動能'}</li>"
        f"</ul>"
        f"<h4>建議</h4>"
        f"<ul>"
        f"<li>20日MA與50日MA顯示趨勢偏{"多頭" if ma20>ma50 else "空頭"}，主趨勢{ "向上" if ma20>ma50 else "向下" }。</li>"
        f"<li>RSI(14)={rsi14:.2f}，{ "過熱" if rsi14>70 else ("超賣" if rsi14<30 else "中性") }，{ "注意可能回檔" if rsi14>70 else ("可能反彈" if rsi14<30 else "趨勢穩定") }。</li>"
        f"<li>MACD柱狀圖={macd_hist:.4f}，{ "多頭動能" if macd_hist>0 else "空頭動能" }較強。</li>"
        f"<li>{ "建議逢低分批買入，並於突破近期高點時加碼；如跌破下方支撐，考慮停損出場。" if ma20>ma50 and macd_hist>0 else ("建議逢高出脫，並於關鍵支撐反彈時再進場。" if ma20<ma50 and macd_hist<0 else "建議維持觀望，待指標趨勢更明確後再行操作。") }</li>"
        f"</ul>"
        f"</div>"
    )

    result = {"plot_img": f"data:image/png;base64,{plot_img}",
              "hist_data": finals.tolist(),
              "commentary_html": commentary_html}
    yield f"data: {json.dumps(result)}\n\n"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stock_stream")
def stock_stream():
    sym = request.args.get("symbol")
    hor = request.args.get("horizon")
    sims = request.args.get("simulations")
    return Response(stream_with_context(simulate_generator(sym, hor, sims)), content_type="text/event-stream")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)