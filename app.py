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
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei','Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

app = Flask(__name__)

HORIZON_MAP = {
    "1D":1, "1W":5, "1M":21, "3M":63,
    "6M":126, "1Y":252, "2Y":504,
    "5Y":1260, "10Y":2520
}

def simulate_generator(symbol, horizon_key, sims):
    # 1. 讀取公司資訊並翻譯
    ticker = yf.Ticker(symbol)
    info   = ticker.info
    ind_en = info.get('industry','N/A')
    sum_en = info.get('longBusinessSummary','')[:4000]
    translator = GoogleTranslator(source='auto', target='zh-TW')
    industry = translator.translate(ind_en) if ind_en!='N/A' else ind_en
    summary  = translator.translate(sum_en) if sum_en else '無可用公司簡介'
    summary  = summary.replace('惠丘市','新竹市')

    # 2. 歷史股價與參數
    hist = ticker.history(period='max', auto_adjust=True)
    current_price = hist['Close'].iloc[-1]
    prices = hist['Close'].values
    rets   = np.diff(prices)/prices[:-1]
    mu     = rets.mean()*252
    sigma  = rets.std(ddof=1)*np.sqrt(252)
    days   = HORIZON_MAP.get(horizon_key,252)
    dt     = 1/252
    sims   = int(sims)

    # 3. 分批模擬並報進度
    chunk = max(1, sims//20)
    paths = []
    for i in range(0, sims, chunk):
        cnt  = min(chunk, sims-i)
        rand = np.random.normal(size=(cnt, days))
        inc  = (mu-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*rand
        sim_p = np.exp(np.cumsum(inc, axis=1))*current_price
        sim_p = np.hstack([np.full((cnt,1),current_price), sim_p])
        paths.append(sim_p)
        pct = int(min((i+cnt)/sims*100,100))
        yield f"data: {pct}\n\n"

    all_paths = np.vstack(paths)   # shape = (sims, days+1)
    finals    = all_paths[:,-1]

    # 4. 統計量
    avg_price = finals.mean()
    min_price = finals.min()
    max_price = finals.max()
    vol       = finals.std()
    vol_pct   = vol/avg_price*100

    # 5. 技術指標
    delta     = hist['Close'].diff()
    gain      = delta.clip(lower=0)
    loss      = -delta.clip(upper=0)
    rs        = gain.rolling(14).mean()/loss.rolling(14).mean()
    rsi14     = 100 - 100/(1+rs.iloc[-1])
    ema12     = hist['Close'].ewm(span=12).mean()
    ema26     = hist['Close'].ewm(span=26).mean()
    macd_hist = (ema12.iloc[-1]-ema26.iloc[-1]) - ((ema12-ema26).ewm(span=9).mean().iloc[-1])
    ma20      = hist['Close'].rolling(20).mean().iloc[-1]
    ma50      = hist['Close'].rolling(50).mean().iloc[-1]

    # 6. 一次輸出 100% 進度
    yield "data: 100\n\n"

    # 7. 繪圖：全部路徑
    x = np.arange(days+1)
    fig,ax = plt.subplots(figsize=(14,7))
    for row in all_paths:
        ax.plot(x, row, lw=0.5, alpha=0.015, color='#007bff')
    ax.plot(x, all_paths.mean(axis=0), lw=4, color='#dc3545', label='平均路徑')
    ax.set_title(f"{symbol} 模擬走勢 {horizon_key} ({sims} 次)", fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel("時間 (天)", fontsize=16)
    ax.set_ylabel("價格 (元)", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=14)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
    plt.close(fig)
    plot_img = base64.b64encode(buf.getvalue()).decode()

    # 8. 建議
    if ma20>ma50 and macd_hist>0 and rsi14<70:
        advice = "建議逢低分批買入，並於突破近期高點時加碼；跌破支撐考慮停損。"
    elif ma20>ma50 and rsi14>=70:
        advice = "多頭趨勢明顯，但 RSI 過熱，建議等待回檔再布局。"
    elif ma20<ma50 and macd_hist<0:
        advice = "趨勢轉空，建議逢反彈至20日均線分批賣出或觀望以控風險。"
    else:
        advice = "訊號混合，建議觀望至指標明確後再操作。"

    # 9. 組合最終結果
    commentary_html = f"""
<div style="font-size:1rem; line-height:1.5;">
  <h4>公司產業與業務</h4>
  <p>屬於 <strong>{industry}</strong> 產業，主要業務：{summary}</p>
  <h4>模擬結果</h4>
  <ul>
    <li>平均：{avg_price:.2f} 元，區間：{min_price:.2f}-{max_price:.2f} 元，波動度：{vol_pct:.2f}%</li>
  </ul>
  <h4>指標解讀</h4>
  <ul>
    <li>MA20={ma20:.2f}, MA50={ma50:.2f} ⇒ 趨勢{'多頭' if ma20>ma50 else '空頭'}</li>
    <li>RSI14={rsi14:.2f} ⇒ {'過熱' if rsi14>70 else ('超賣' if rsi14<30 else '中性')}</li>
    <li>MACD Hist={macd_hist:.4f} ⇒ {'多頭動能' if macd_hist>0 else '空頭動能'}</li>
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
    return Response(
        stream_with_context(
            simulate_generator(
                request.args.get("symbol"),
                request.args.get("horizon"),
                request.args.get("simulations")
            )
        ),
        content_type="text/event-stream"
    )

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)
