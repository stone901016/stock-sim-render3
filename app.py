from flask import Flask, render_template, request, Response, stream_with_context
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator
import io, base64, json

# 字體設定
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei','Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

app = Flask(__name__)

HORIZON_MAP = {"1D":1,"1W":5,"1M":21,"3M":63,"6M":126,"1Y":252,"2Y":504,"5Y":1260,"10Y":2520}

def simulate_generator(symbol, horizon_key, sims):
    # 翻譯
    ticker = yf.Ticker(symbol)
    info = ticker.info
    en_ind = info.get('industry','N/A')
    en_sum = info.get('longBusinessSummary','無可用公司簡介')
    tr = GoogleTranslator(source='auto',target='zh-TW')
    industry = tr.translate(en_ind) if en_ind!='N/A' else 'N/A'
    summary = tr.translate(en_sum[:4000]) if en_sum else '無可用公司簡介'
    summary = summary.replace('惠丘市','新竹市')

    # 歷史資料
    hist = ticker.history(period="max",auto_adjust=True)
    prices = hist['Close'].values
    curr = prices[-1]
    rets = np.diff(prices)/prices[:-1]
    mu = rets.mean()*252
    sigma = rets.std(ddof=1)*np.sqrt(252)
    days = HORIZON_MAP.get(horizon_key,252)
    dt = 1/252
    sims = int(sims)
    chunk = max(1,sims//100)
    parts = []

    # SSE 進度
    for i in range(0,sims,chunk):
        cnt = min(chunk, sims-i)
        z = np.random.normal(size=(cnt,days))
        inc = (mu-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z
        S = curr * np.exp(np.cumsum(inc,axis=1))
        S = np.concatenate([np.full((cnt,1),curr),S],axis=1)
        parts.append(S)
        p = int(min((i+cnt)/sims*100,100))
        yield f"data:{p}\n\n"

    all_paths = np.vstack(parts)
    finals = all_paths[:,-1]
    avg_p, min_p, max_p = finals.mean(), finals.min(), finals.max()
    vol = finals.std(); vol_pct = vol/avg_p*100

    # 指標
    d = hist['Close'].diff()
    g = d.clip(lower=0); l = -d.clip(upper=0)
    ag = g.rolling(14).mean(); al = l.rolling(14).mean()
    rs = ag/al
    rsi14 = 100-100/(1+rs.iloc[-1])
    ema12 = hist['Close'].ewm(span=12).mean()
    ema26 = hist['Close'].ewm(span=26).mean()
    macd_hist = (ema12.iloc[-1]-ema26.iloc[-1]) - ( (ema12-ema26).ewm(span=9).mean().iloc[-1] )
    ma20 = hist['Close'].rolling(20).mean().iloc[-1]
    ma50 = hist['Close'].rolling(50).mean().iloc[-1]

    # 走勢圖（前2000條 + 平均線）
    sample_n = min(2000, all_paths.shape[0])
    x = np.arange(days+1)
    fig,ax = plt.subplots(figsize=(12,6))
    for path in all_paths[:sample_n]:
        ax.plot(x,path,alpha=0.02,lw=0.5,color='#007bff')
    ax.plot(x,all_paths.mean(axis=0),lw=2,color='#dc3545',label='平均路徑')
    ax.set_title(f"{symbol} 預測走勢 ({horizon_key}, {sims:,} 次模擬)",fontsize=16,fontweight='bold')
    ax.set_xlabel("天數 (天)"); ax.set_ylabel("價格 (元)")
    ax.legend()
    buf=io.BytesIO(); fig.tight_layout(); fig.savefig(buf,format='png'); plt.close(fig)
    trend_img = base64.b64encode(buf.getvalue()).decode()

    # 次數分佈
    fig,ax=plt.subplots(figsize=(6,4))
    ax.hist(finals,bins=50,color='#28a745',alpha=0.7)
    ax.set_title("最終價格分佈")
    ax.set_xlabel("價格 (元)"); ax.set_ylabel("次數")
    buf2=io.BytesIO(); fig.tight_layout(); fig.savefig(buf2,format='png'); plt.close(fig)
    hist_img = base64.b64encode(buf2.getvalue()).decode()

    # 建議
    if ma20>ma50 and macd_hist>0 and rsi14<70:
        advice="趨勢偏多，建議分批逢低買入並持有。"
    elif ma20>ma50 and rsi14>=70:
        advice="多頭趨勢，但 RSI 過熱，建議待回檔再佈局。"
    elif ma20<ma50 and macd_hist<0:
        advice="趨勢轉空，建議逢高賣出或觀望。"
    else:
        advice="訊號混合，建議先觀望，待訊號明確後操作。"

    # 結果
    res = {
      "trend_img":f"data:image/png;base64,{trend_img}",
      "hist_img": f"data:image/png;base64,{hist_img}",
      "stats":{"avg":f"{avg_p:.2f}","curr":f"{curr:.2f}",
               "min":f"{min_p:.2f}","max":f"{max_p:.2f}",
               "vol":f"{vol:.2f}","vol_pct":f"{vol_pct:.2f}%"},
      "inds":{"MA20":f"{ma20:.2f}","MA50":f"{ma50:.2f}",
              "RSI14":f"{rsi14:.2f}","MACD":f"{macd_hist:.4f}"},
      "industry":industry,"summary":summary,"advice":advice
    }
    yield f"data:{json.dumps(res)}\n\n"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stock_stream")
def stock_stream():
    return Response(stream_with_context(
      simulate_generator(
        request.args["symbol"],
        request.args["horizon"],
        request.args["simulations"]
      )
    ),content_type="text/event-stream")

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)
