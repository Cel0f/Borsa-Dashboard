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
.main { background-color: #010409; }
.block-container { padding: 1rem 2rem; }
.stButton>button {
    background: #1a6630; color: #6ee89a; border: none;
    border-radius: 8px; padding: 10px 30px; font-size: 15px; font-weight: 600;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

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
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=p, adjust=False).mean()
    up = high.diff()
    dn = -low.diff()
    dip = 100 * up.where((up > dn) & (up > 0), 0.0).ewm(span=p, adjust=False).mean() / atr
    din = 100 * dn.where((dn > up) & (dn > 0), 0.0).ewm(span=p, adjust=False).mean() / atr
    dx = 100 * (dip - din).abs() / (dip + din).replace(0, np.nan)
    return dx.ewm(span=p, adjust=False).mean()

def calc_stoch(high, low, close, k=14):
    lo = low.rolling(k).min()
    hi = high.rolling(k).max()
    return 100 * (close - lo) / (hi - lo).replace(0, np.nan)

def calc_bb(close, p=20, s=2):
    mid = close.rolling(p).mean()
    sig = close.rolling(p).std()
    return mid + s * sig, mid, mid - s * sig

def calc_atr(high, low, close, p=14):
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

def karar(r, ml, ms):
    if r > 55 and ml > ms:
        return "AL"
    if r < 45 and ml < ms:
        return "SAT"
    return "NOTR"

def karar_df(df):
    if df is None or len(df) < 26:
        return "NOTR"
    try:
        c = df["Close"].squeeze().astype(float)
        r = float(calc_rsi(c).iloc[-1])
        ml, ms = calc_macd(c)
        return karar(r, float(ml.iloc[-1]), float(ms.iloc[-1]))
    except:
        return "NOTR"

def skor_hesapla(df):
    if df is None or len(df) < 30:
        return None
    try:
        c = df["Close"].squeeze().astype(float)
        h = df["High"].squeeze().astype(float)
        l = df["Low"].squeeze().astype(float)
        v = df["Volume"].squeeze().astype(float)
        r = float(calc_rsi(c).iloc[-1])
        ml, ms = calc_macd(c)
        e20 = float(calc_ema(c, 20).iloc[-1])
        e50 = float(calc_ema(c, 50).iloc[-1])
        e200 = float(calc_ema(c, 200).iloc[-1]) if len(c) >= 200 else float(calc_ema(c, 50).iloc[-1])
        adxv = float(calc_adx(h, l, c).iloc[-1])
        stk = float(calc_stoch(h, l, c).iloc[-1])
        atrv = float(calc_atr(h, l, c).iloc[-1])
        cl = float(c.iloc[-1])
        bu, bm, bl = calc_bb(c)
        bw = float((bu - bl).iloc[-1] / bm.iloc[-1]) if float(bm.iloc[-1]) != 0 else 0
        bwp = float((bu - bl).iloc[-2] / bm.iloc[-2]) if float(bm.iloc[-2]) != 0 else bw
        vr = float(v.iloc[-1]) / float(v.rolling(20).mean().iloc[-1])
        stop = round(cl - atrv * 2.0, 2)
        # Gunluk momentum skoru (gun hissesi icin)
        gun_skor = 0
        if 50 < r < 70: gun_skor += 2      # RSI ideal bolgede
        if stk < 80: gun_skor += 1          # Stoch asiri alimda degil
        if ml.iloc[-1] > ms.iloc[-1]: gun_skor += 2  # MACD pozitif
        if adxv > 25: gun_skor += 2         # Guclu trend
        if vr > 1.5: gun_skor += 2          # Yuksek hacim
        if cl > e20: gun_skor += 1          # Fiyat EMA20 ustunde
        s = 0
        if r < 30: s += 2
        elif r < 55: s += 1
        if ml.iloc[-1] > ms.iloc[-1]: s += 1
        if e20 > e50 and e50 > e200: s += 2
        elif cl > e200: s += 1
        if adxv > 25: s += 1
        if stk < 20: s += 1
        if bwp < 0.05 and bw >= 0.05: s += 1
        if vr > 1.2: s += 1
        return {
            "RSI": round(r, 1), "Stoch": round(stk, 1),
            "MACD": round(float(ml.iloc[-1] - ms.iloc[-1]), 2),
            "ADX": round(adxv, 1), "ATR": round(atrv, 2),
            "Stop": stop, "VolR": round(vr, 2),
            "Skor": s, "GunSkor": gun_skor,
            "K1G": karar(r, float(ml.iloc[-1]), float(ms.iloc[-1]))
        }
    except:
        return None

@st.cache_data(ttl=900)
def tara_tum(liste, suffix=""):
    sonuc = []
    for sembol in liste:
        try:
            dg = yf.download(sembol + suffix, period="2y", interval="1d", progress=False, auto_adjust=True)
            dh = yf.download(sembol + suffix, period="5y", interval="1wk", progress=False, auto_adjust=True)
            da = yf.download(sembol + suffix, period="10y", interval="1mo", progress=False, auto_adjust=True)
            if dg is None or len(dg) < 30:
                continue
            t = skor_hesapla(dg)
            if t is None:
                continue
            k1h = karar_df(dh)
            k1a = karar_df(da)
            al_puan = sum(1 for k in [t["K1G"], k1h, k1a] if k == "AL")
            sonuc.append({
                "Hisse": sembol, "Skor": t["Skor"], "GunSkor": t["GunSkor"],
                "RSI": t["RSI"], "Stoch": t["Stoch"], "MACD": t["MACD"],
                "ADX": t["ADX"], "ATR": t["ATR"], "Stop": t["Stop"],
                "VolR": t["VolR"], "1G": t["K1G"], "1H": k1h, "1A": k1a,
                "AL_Puan": al_puan
            })
        except:
            pass
    return pd.DataFrame(sonuc)

def filtrele_ve_sirala(df):
    # Haftalik: RSI<70, Stoch<80, sirala
    haftalik = df[(df["RSI"] < 70) & (df["Stoch"] < 80)].copy()
    haftalik = haftalik.sort_values(["AL_Puan", "Skor"], ascending=False).reset_index(drop=True)
    # Gunluk: RSI 45-72, Stoch<85, ADX>20, MACD pozitif, sirala
    gunluk = df[
        (df["RSI"] > 45) & (df["RSI"] < 72) &
        (df["Stoch"] < 85) &
        (df["ADX"] > 20) &
        (df["MACD"] > 0) &
        (df["1G"] == "AL")
    ].copy()
    gunluk = gunluk.sort_values("GunSkor", ascending=False).reset_index(drop=True)
    return haftalik, gunluk

def badge(v):
    if v == "AL":
        return "<span style='background:#0d3320;color:#6ee89a;padding:2px 9px;border-radius:4px;font-size:11px;font-weight:700'>AL</span>"
    if v == "SAT":
        return "<span style='background:#3d0c0c;color:#f08080;padding:2px 9px;border-radius:4px;font-size:11px;font-weight:700'>SAT</span>"
    return "<span style='background:#3d2e00;color:#f0c060;padding:2px 9px;border-radius:4px;font-size:11px;font-weight:700'>NOTR</span>"

def skor_html(v):
    c = "#1a9e4a" if v >= 7 else "#e08800" if v >= 5 else "#c0392b"
    return "<span style='color:" + c + ";font-weight:700;font-size:13px'>" + str(v) + "/9</span>"

def rsi_html(v):
    c = "#f08080" if v > 70 else "#6ee89a" if v < 30 else "#ccc"
    return "<span style='color:" + c + "'>" + str(v) + "</span>"

def tablo_html(df, n, gun_hisse="", hafta_hisseler=None):
    if hafta_hisseler is None:
        hafta_hisseler = []
    satirlar = ""
    for i, r in df.head(n).iterrows():
        is_gun = r["Hisse"] == gun_hisse
        is_hafta = r["Hisse"] in hafta_hisseler
        if is_gun:
            bg = "#2d1f00"
            left_border = "border-left:4px solid #ffd700;"
            hisse_label = r["Hisse"] + " ⚡"
            hisse_color = "#ffd700"
        elif is_hafta:
            bg = "#0d2040"
            left_border = "border-left:4px solid #4da6ff;"
            hisse_label = r["Hisse"] + " ★"
            hisse_color = "#4da6ff"
        else:
            bg = "#111"
            left_border = ""
            hisse_label = r["Hisse"]
            hisse_color = "#fff"
        macd_c = "#6ee89a" if r["MACD"] > 0 else "#f08080"
        adx_c = "#6ee89a" if r["ADX"] > 25 else "#aaa"
        satirlar += (
            "<tr style='background:" + bg + ";border-bottom:1px solid #222;" + left_border + "'>"
            "<td style='padding:8px 10px;font-weight:700;color:" + hisse_color + "'>" + hisse_label + "</td>"
            "<td style='text-align:center'>" + skor_html(r["Skor"]) + "</td>"
            "<td style='text-align:center'>" + rsi_html(r["RSI"]) + "</td>"
            "<td style='text-align:center;color:#ccc'>" + str(r["Stoch"]) + "</td>"
            "<td style='text-align:center;color:" + macd_c + "'>" + str(r["MACD"]) + "</td>"
            "<td style='text-align:center;color:" + adx_c + "'>" + str(r["ADX"]) + "</td>"
            "<td style='text-align:center;color:#e08800'>" + str(r["ATR"]) + "</td>"
            "<td style='text-align:center;color:#f08080'>" + str(r["Stop"]) + "</td>"
            "<td style='text-align:center'>" + badge(r["1G"]) + "</td>"
            "<td style='text-align:center'>" + badge(r["1H"]) + "</td>"
            "<td style='text-align:center'>" + badge(r["1A"]) + "</td>"
            "</tr>"
        )
    return (
        "<div style='overflow-x:auto'>"
        "<table style='width:100%;border-collapse:collapse;font-size:12px;font-family:monospace'>"
        "<thead><tr style='color:#8b949e;border-bottom:2px solid #30363d;background:#0d1117'>"
        "<th style='padding:8px 10px;text-align:left'>Hisse</th>"
        "<th style='padding:8px'>Skor</th>"
        "<th style='padding:8px'>RSI</th>"
        "<th style='padding:8px'>Stoch</th>"
        "<th style='padding:8px'>MACD</th>"
        "<th style='padding:8px'>ADX</th>"
        "<th style='padding:8px'>ATR</th>"
        "<th style='padding:8px'>Stop</th>"
        "<th style='padding:8px'>1G</th>"
        "<th style='padding:8px'>1H</th>"
        "<th style='padding:8px'>1A</th>"
        "</tr></thead>"
        "<tbody>" + satirlar + "</tbody>"
        "</table></div>"
    )

def highlight_box(emoji, baslik, hisse, aciklama, renk):
    return (
        "<div style='background:" + renk + ";border-radius:10px;padding:14px 18px;margin-bottom:8px;border-left:5px solid " + renk.replace("1a","ff").replace("0d","88") + "'>"
        "<div style='font-size:20px;font-weight:800;color:#fff'>" + emoji + "  " + hisse + "</div>"
        "<div style='font-size:13px;font-weight:600;color:#ddd;margin-top:2px'>" + baslik + "</div>"
        "<div style='font-size:12px;color:#aaa;margin-top:4px'>" + aciklama + "</div>"
        "</div>"
    )

# ── SAYFA ─────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center;color:#fff;margin-bottom:4px'>📊 Borsa Dashboard</h1>", unsafe_allow_html=True)
saat = datetime.now().strftime("%d.%m.%Y %H:%M")
st.markdown(
    "<p style='text-align:center;color:#8b949e;font-size:13px'>Guncelleme: " + saat +
    " &nbsp;|&nbsp; Filtre: RSI<70, Stoch<80 &nbsp;|&nbsp; ~15 dk gecikmeli</p>",
    unsafe_allow_html=True
)

col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
with col_btn2:
    yenile = st.button("Yenile")
if yenile:
    st.cache_data.clear()

with st.spinner("BIST100 ve S&P500 taraniyor... (3-5 dakika)"):
    df_bist_raw = tara_tum(BIST100, ".IS")
    df_sp_raw = tara_tum(SP500, "")

df_bist, df_bist_gun = filtrele_ve_sirala(df_bist_raw)
df_sp, df_sp_gun = filtrele_ve_sirala(df_sp_raw)

# Gunun ve haftanin hisseleri
bist_gun_hisse = df_bist_gun["Hisse"].iloc[0] if len(df_bist_gun) > 0 else ""
bist_hafta_hisseler = df_bist.head(2)["Hisse"].tolist() if len(df_bist) >= 2 else []
sp_gun_hisse = df_sp_gun["Hisse"].iloc[0] if len(df_sp_gun) > 0 else ""
sp_hafta_hisseler = df_sp.head(2)["Hisse"].tolist() if len(df_sp) >= 2 else []

st.markdown("---")

# Highlight kutulari
col_h1, col_h2 = st.columns(2)

with col_h1:
    st.markdown("<div style='margin-bottom:8px'>", unsafe_allow_html=True)
    if bist_gun_hisse:
        gun_row = df_bist_gun.iloc[0]
        st.markdown(highlight_box(
            "⚡", "GUNUN BIST HISSESI — Gunluk islem icin",
            bist_gun_hisse,
            "RSI: " + str(gun_row["RSI"]) + "  |  ADX: " + str(gun_row["ADX"]) + "  |  Hacim: " + str(gun_row["VolR"]) + "x  |  Stop: " + str(gun_row["Stop"]),
            "#2d1f00"
        ), unsafe_allow_html=True)
    if len(bist_hafta_hisseler) >= 1:
        r1 = df_bist[df_bist["Hisse"] == bist_hafta_hisseler[0]].iloc[0]
        st.markdown(highlight_box(
            "★", "HAFTANIN 1. BIST HISSESI — Haftalik islem icin",
            bist_hafta_hisseler[0],
            "Skor: " + str(r1["Skor"]) + "/9  |  RSI: " + str(r1["RSI"]) + "  |  1G/1H/1A: " + r1["1G"] + "/" + r1["1H"] + "/" + r1["1A"],
            "#0d2040"
        ), unsafe_allow_html=True)
    if len(bist_hafta_hisseler) >= 2:
        r2 = df_bist[df_bist["Hisse"] == bist_hafta_hisseler[1]].iloc[0]
        st.markdown(highlight_box(
            "★", "HAFTANIN 2. BIST HISSESI — Haftalik islem icin",
            bist_hafta_hisseler[1],
            "Skor: " + str(r2["Skor"]) + "/9  |  RSI: " + str(r2["RSI"]) + "  |  1G/1H/1A: " + r2["1G"] + "/" + r2["1H"] + "/" + r2["1A"],
            "#0d2040"
        ), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_h2:
    st.markdown("<div style='margin-bottom:8px'>", unsafe_allow_html=True)
    if sp_gun_hisse:
        gun_row_sp = df_sp_gun.iloc[0]
        st.markdown(highlight_box(
            "⚡", "GUNUN S&P500 HISSESI — Gunluk islem icin",
            sp_gun_hisse,
            "RSI: " + str(gun_row_sp["RSI"]) + "  |  ADX: " + str(gun_row_sp["ADX"]) + "  |  Hacim: " + str(gun_row_sp["VolR"]) + "x  |  Stop: " + str(gun_row_sp["Stop"]),
            "#2d1f00"
        ), unsafe_allow_html=True)
    if len(sp_hafta_hisseler) >= 1:
        r1 = df_sp[df_sp["Hisse"] == sp_hafta_hisseler[0]].iloc[0]
        st.markdown(highlight_box(
            "★", "HAFTANIN 1. S&P500 HISSESI — Haftalik islem icin",
            sp_hafta_hisseler[0],
            "Skor: " + str(r1["Skor"]) + "/9  |  RSI: " + str(r1["RSI"]) + "  |  1G/1H/1A: " + r1["1G"] + "/" + r1["1H"] + "/" + r1["1A"],
            "#0d2040"
        ), unsafe_allow_html=True)
    if len(sp_hafta_hisseler) >= 2:
        r2 = df_sp[df_sp["Hisse"] == sp_hafta_hisseler[1]].iloc[0]
        st.markdown(highlight_box(
            "★", "HAFTANIN 2. S&P500 HISSESI — Haftalik islem icin",
            sp_hafta_hisseler[1],
            "Skor: " + str(r2["Skor"]) + "/9  |  RSI: " + str(r2["RSI"]) + "  |  1G/1H/1A: " + r2["1G"] + "/" + r2["1H"] + "/" + r2["1A"],
            "#0d2040"
        ), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### BIST100 - Top 10")
    st.markdown(
        "<div style='font-size:11px;color:#8b949e;margin-bottom:8px'>"
        "⚡ Gunun hissesi &nbsp;|&nbsp; ★ Haftanin hisseleri &nbsp;|&nbsp; Filtre: RSI &lt; 70, Stoch &lt; 80"
        "</div>", unsafe_allow_html=True
    )
    st.markdown(
        tablo_html(df_bist, 10, bist_gun_hisse, bist_hafta_hisseler),
        unsafe_allow_html=True
    )
    with st.expander("Tam liste"):
        st.markdown(tablo_html(df_bist, len(df_bist), bist_gun_hisse, bist_hafta_hisseler), unsafe_allow_html=True)

with col2:
    st.markdown("### S&P500 - Top 10")
    st.markdown(
        "<div style='font-size:11px;color:#8b949e;margin-bottom:8px'>"
        "⚡ Gunun hissesi &nbsp;|&nbsp; ★ Haftanin hisseleri &nbsp;|&nbsp; Filtre: RSI &lt; 70, Stoch &lt; 80"
        "</div>", unsafe_allow_html=True
    )
    st.markdown(
        tablo_html(df_sp, 10, sp_gun_hisse, sp_hafta_hisseler),
        unsafe_allow_html=True
    )
    with st.expander("Tam liste"):
        st.markdown(tablo_html(df_sp, len(df_sp), sp_gun_hisse, sp_hafta_hisseler), unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#484f58;font-size:11px'>Teknik analiz amaclidir, yatirim tavsiyesi degildir. Veri: Yahoo Finance</p>",
    unsafe_allow_html=True
)
