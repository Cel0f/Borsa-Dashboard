import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Borsa Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
body { background-color: #010409; }
.main { background-color: #010409; }
.block-container { padding: 1rem 2rem; }
h1 { color: #fff; text-align: center; }
.stButton>button {
    background: #1a6630; color: #6ee89a; border: none;
    border-radius: 8px; padding: 10px 30px; font-size: 15px; font-weight: 600;
    width: 100%; cursor: pointer;
}
.stButton>button:hover { background: #0d3320; }
</style>
""", unsafe_allow_html=True)

# ── LİSTELER ──────────────────────────────────────────────
BIST100 = [
    "AKBNK","GARAN","ISCTR","YKBNK","HALKB","VAKBN","TCELL","TUPRS","KCHOL","SAHOL",
    "EREGL","PETKM","BIMAS","MGROS","ASELS","THYAO","PGSUS","SISE","TOASO","FROTO",
    "ARCLK","VESTL","TTKOM","DOHOL","ENKAI","KRDMD","CCOLA","AEFES","ULKER","TATGD",
    "ODAS","AKSEN","ZOREN","AKENR","AYDEM","AYGAZ","DOAS","OTKAR","BRSAN","ISGYO",
    "EKGYO","TRGYO","MPARK","KOZAL","GLYHO","LOGO","NETAS","INDES","KAREL","ALKIM",
    "GUBRF","BAGFS","CIMSA","AKCNS","BOLUC","GOLTS","ADANA","UNYEC","TRKCM","ANACM",
    "SODA","HEKTS","YATAS","MAVI","PRKME","TMSN","ULUSE","VKGYO","ALBRK","ISDMR",
    "AGESA","ANHYT","AVOD","BAKAB","BANVT","BRMEN","BUCIM","CEMTS","CLEBI","DESA",
    "DEVA","EGEEN","IZMDC","KARSN","KLNMA","IPEKE","SDTTR","BERA","BTCIM","CWENE",
    "EBEBK","EGGUB","FENER","GESAN","GMTAS","GSDHO","KONTR","OYAKC","SMRTG","TNZTP"
]

SP500 = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","V",
    "UNH","XOM","MA","LLY","JNJ","PG","HD","MRK","AVGO","CVX",
    "ABBV","COST","PEP","KO","BAC","WMT","ADBE","CRM","ACN","TMO",
    "MCD","CSCO","ABT","DHR","TXN","NEE","PM","LIN","NFLX","AMD",
    "ORCL","QCOM","UPS","RTX","HON","LOW","AMGN","IBM","CAT","GS",
    "BA","SBUX","INTU","ISRG","SPGI","BLK","MS","AXP","T","DE",
    "GILD","ADI","REGN","VRTX","MDLZ","C","PLD","SYK","CB","ZTS",
    "SCHW","SO","DUK","MO","TGT","MMM","F","GM","USB","GE",
    "WFC","COP","SLB","EOG","PSA","EQR","AMT","EQIX","DLR","SPG",
    "NXPI","MCHP","KLAC","LRCX","AMAT","MU","PANW","CRWD","SNOW","PLTR"
]

# ── GÖSTERGELER ───────────────────────────────────────────
def calc_rsi(close, period=14):
    delta = close.diff()
    ag = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    al = (-delta.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))

def calc_macd(close, fast=12, slow=26, signal=9):
    ml = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    return ml, ml.ewm(span=signal, adjust=False).mean()

def calc_ema(close, p):
    return close.ewm(span=p, adjust=False).mean()

def calc_adx(high, low, close, p=14):
    tr  = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=p, adjust=False).mean()
    up = high.diff(); dn = -low.diff()
    dip = 100 * up.where((up>dn)&(up>0), 0.0).ewm(span=p, adjust=False).mean() / atr
    din = 100 * dn.where((dn>up)&(dn>0), 0.0).ewm(span=p, adjust=False).mean() / atr
    dx  = 100 * (dip-din).abs() / (dip+din).replace(0, np.nan)
    return dx.ewm(span=p, adjust=False).mean()

def calc_stoch(high, low, close, k=14):
    lo = low.rolling(k).min(); hi = high.rolling(k).max()
    return 100 * (close - lo) / (hi - lo).replace(0, np.nan)

def calc_bb(close, p=20, s=2):
    mid = close.rolling(p).mean(); sig = close.rolling(p).std()
    return mid+s*sig, mid, mid-s*sig

def calc_atr(high, low, close, p=14):
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

def karar(r, ml, ms):
    if r > 55 and ml > ms:   return "AL"
    if r < 45 and ml < ms:   return "SAT"
    return "NOTR"

def karar_df(df):
    if df is None or len(df) < 26: return "NOTR"
    try:
        c = df["Close"].squeeze().astype(float)
        r = float(calc_rsi(c).iloc[-1])
        ml, ms = calc_macd(c)
        return karar(r, float(ml.iloc[-1]), float(ms.iloc[-1]))
    except: return "NOTR"

def skor_hesapla(df):
    if df is None or len(df) < 30: return None
    try:
        c  = df["Close"].squeeze().astype(float)
        h  = df["High"].squeeze().astype(float)
        l  = df["Low"].squeeze().astype(float)
        v  = df["Volume"].squeeze().astype(float)

        r      = float(calc_rsi(c).iloc[-1])
        ml, ms = calc_macd(c)
        e20    = float(calc_ema(c, 20).iloc[-1])
        e50    = float(calc_ema(c, 50).iloc[-1])
        e200   = float(calc_ema(c, 200).iloc[-1]) if len(c) >= 200 else float(calc_ema(c, 50).iloc[-1])
        adxv   = float(calc_adx(h, l, c).iloc[-1])
        stk    = float(calc_stoch(h, l, c).iloc[-1])
        atrv   = float(calc_atr(h, l, c).iloc[-1])
        cl     = float(c.iloc[-1])
        bu, bm, bl = calc_bb(c)
        bw     = float((bu-bl).iloc[-1] / bm.iloc[-1]) if float(bm.iloc[-1]) != 0 else 0
        bwp    = float((bu-bl).iloc[-2] / bm.iloc[-2]) if float(bm.iloc[-2]) != 0 else bw
        vr     = float(v.iloc[-1]) / float(v.rolling(20).mean().iloc[-1])
        stop   = round(cl - atrv * 2.0, 2)

        # ── SKOR (9 üzerinden — Pine Script ile aynı) ────
        s = 0
        if r < 30:                      s += 2  # RSI aşırı satım
        elif r < 55:                    s += 1  # RSI normal
        if ml.iloc[-1] > ms.iloc[-1]:  s += 1  # MACD pozitif
        if e20 > e50 and e50 > e200:   s += 2  # EMA tam sıralı
        elif cl > e200:                 s += 1  # Fiyat EMA200 üstünde
        if adxv > 25:                   s += 1  # Güçlü trend
        if stk < 20:                    s += 1  # Stoch aşırı satım
        if bwp < 0.05 and bw >= 0.05:  s += 1  # BB sıkışma kırılımı
        if vr > 1.2:                    s += 1  # Hacim artışı

        return {
            "RSI":   round(r, 1),
            "Stoch": round(stk, 1),
            "MACD":  round(float(ml.iloc[-1] - ms.iloc[-1]), 2),
            "ADX":   round(adxv, 1),
            "ATR":   round(atrv, 2),
            "Stop":  stop,
            "BB_W":  round(bw, 3),
            "VolR":  round(vr, 2),
            "EMA20": round(e20, 2),
            "EMA50": round(e50, 2),
            "EMA200":round(e200, 2),
            "Skor":  s,
            "K1G":   karar(r, float(ml.iloc[-1]), float(ms.iloc[-1]))
        }
    except: return None

@st.cache_data(ttl=900)  # 15 dk cache
def tara_tum(liste, suffix=""):
    sonuc = []
    for sembol in liste:
        try:
            dg = yf.download(sembol+suffix, period="2y",  interval="1d",  progress=False, auto_adjust=True)
            dh = yf.download(sembol+suffix, period="5y",  interval="1wk", progress=False, auto_adjust=True)
            da = yf.download(sembol+suffix, period="10y", interval="1mo", progress=False, auto_adjust=True)
            if dg is None or len(dg) < 30: continue
            t = skor_hesapla(dg)
            if t is None: continue
            k1h = karar_df(dh)
            k1a = karar_df(da)
            al_puan = sum(1 for k in [t["K1G"], k1h, k1a] if k == "AL")
            sonuc.append({
                "Hisse": sembol, "Skor": t["Skor"], "RSI": t["RSI"],
                "Stoch": t["Stoch"], "MACD": t["MACD"], "ADX": t["ADX"],
                "ATR": t["ATR"], "Stop": t["Stop"], "BB_W": t["BB_W"],
                "VolR": t["VolR"], "EMA20": t["EMA20"], "EMA50": t["EMA50"],
                "EMA200": t["EMA200"], "1G": t["K1G"], "1H": k1h, "1A": k1a,
                "AL_Puan": al_puan
            })
        except: pass
    return pd.DataFrame(sonuc).sort_values(["AL_Puan","Skor"], ascending=False).reset_index(drop=True)

# ── BADGE HTML ────────────────────────────────────────────
def badge(v):
    renkler = {"AL": ("#0d3320","#6ee89a"), "SAT": ("#3d0c0c","#f08080"), "NOTR": ("#3d2e00","#f0c060")}
    bg, fg = renkler.get(v, ("#1a1a2e","#ccc"))
    return f'<span style="background:{bg};color:{fg};padding:2px 9px;border-radius:4px;font-size:11px;font-weight:700">{v}</span>'

def skor_html(v):
    c = "#1a9e4a" if v>=7 else "#e08800" if v>=5 else "#c0392b"
    return f'<span style="color:{c};font-weight:700;font-size:13px">{v}/9</span>'

def rsi_html(v):
    c = "#f08080" if v>70 else "#6ee89a" if v<30 else "#ccc"
    return f'<span style="color:{c}">{v}</span>'

def tablo_html(df, n=10):
    satirlar = ""
    for i, r in df.head(n).iterrows():
        bg = "#0d1f0d" if r["AL_Puan"]==3 else "#111"
        satirlar += f"""<tr style="background:{bg};border-bottom:1px solid #222">
          <td style="padding:8px 10px;font-weight:600;color:#fff">{r["Hisse"]}</td>
          <td style="text-align:center">{skor_html(r["Skor"])}</td>
          <td style="text-align:center">{rsi_html(r["RSI"])}</td>
          <td style="text-align:center;color:#ccc">{r["Stoch"]}</td>
          <td style="text-align:center;color:{"#6ee89a" if r["MACD"]>0 else "#f08080"}">{r["MACD"]}</td>
          <td style="text-align:center;color:{"#6ee89a" if r["ADX"]>25 else "#aaa"}">{r["ADX"]}</td>
          <td style="text-align:center;color:#e08800">{r["ATR"]}</td>
          <td style="text-align:center;color:#f08080">{r["Stop"]}</td>
          <td style="text-align:center">{badge(r["1G"])}</td>
          <td style="text-align:center">{badge(r["1H"])}</td>
          <td style="text-align:center">{badge(r["1A"])}</td>
        </tr>"""
    return f"""<div style="overflow-x:auto">
    <table style="width:100%;border-collapse:collapse;font-size:12px;font-family:monospace">
      <thead><tr style="color:#8b949e;border-bottom:2px solid #30363d;background:#0d1117">
        <th style="padding:8px 10px;text-align:left">Hisse</th>
        <th style="padding:8px">Skor</th>
        <th style="padding:8px">RSI</th>
        <th style="padding:8px">Stoch</th>
        <th style="padding:8px">MACD</th>
        <th style="padding:8px">ADX</th>
        <th style="padding:8px">ATR</th>
        <th style="padding:8px">Stop</th>
        <th style="padding:8px">1G</th>
        <th style="padding:8px">1H</th>
        <th style="padding:8px">1A</th>
      </tr></thead>
      <tbody>{satirlar}</tbody>
    </table></div>"""

# ── SAYFA ─────────────────────────────────────────────────
st.markdown("<h1>📊 Borsa Dashboard</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;color:#8b949e;font-size:13px'>Sıralama: AL puanı + Teknik Skor (9 üzerinden) &nbsp;|&nbsp; Güncelleme: {datetime.now().strftime(\'%d.%m.%Y %H:%M\')} &nbsp;|&nbsp; ~15 dk gecikmeli</p>", unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([2,1,2])
with col_btn2:
    yenile = st.button("🔄 Yenile")

if yenile:
    st.cache_data.clear()

with st.spinner("BIST100 ve S&P500 taranıyor... (3-5 dakika)"):
    df_bist = tara_tum(BIST100, ".IS")
    df_sp   = tara_tum(SP500, "")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🇹🇷 BIST100 — Top 10")
    st.markdown(f"*{len(df_bist)} hisse tarandı*")
    st.markdown(tablo_html(df_bist, 10), unsafe_allow_html=True)
    with st.expander("📋 Tam liste"):
        st.markdown(tablo_html(df_bist, len(df_bist)), unsafe_allow_html=True)

with col2:
    st.markdown("### 🇺🇸 S&P500 — Top 10")
    st.markdown(f"*{len(df_sp)} hisse tarandı*")
    st.markdown(tablo_html(df_sp, 10), unsafe_allow_html=True)
    with st.expander("📋 Tam liste"):
        st.markdown(tablo_html(df_sp, len(df_sp)), unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#484f58;font-size:11px'>Teknik analiz amaçlıdır, yatırım tavsiyesi değildir. Veri: Yahoo Finance</p>", unsafe_allow_html=True)
