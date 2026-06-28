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
    up = high.diff(); dn = -low.diff()
    dip = 100 * up.where((up > dn) & (up > 0), 0.0).ewm(span=p, adjust=False).mean() / atr
    din = 100 * dn.where((dn > up) & (dn > 0), 0.0).ewm(span=p, adjust=False).mean() / atr
    dx = 100 * (dip - din).abs() / (dip + din).replace(0, np.nan)
    return dx.ewm(span=p, adjust=False).mean()

def calc_stoch(high, low, close, k=14):
    lo = low.rolling(k).min(); hi = high.rolling(k).max()
    return 100 * (close - lo) / (hi - lo).replace(0, np.nan)

def calc_bb(close, p=20, s=2):
    mid = close.rolling(p).mean(); sig = close.rolling(p).std()
    return mid + s * sig, mid, mid - s * sig

def calc_atr(high, low, close, p=14):
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

def calc_aroon(high, low, period=14):
    aroon_up = high.rolling(period+1).apply(lambda x: (x.argmax() / period) * 100, raw=True)
    aroon_dn = low.rolling(period+1).apply(lambda x: (x.argmin() / period) * 100, raw=True)
    return aroon_up, aroon_dn, aroon_up - aroon_dn

def calc_obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def karar(r, ml, ms):
    if r > 55 and ml > ms: return "AL"
    if r < 45 and ml < ms: return "SAT"
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
        c = df["Close"].squeeze().astype(float)
        h = df["High"].squeeze().astype(float)
        l = df["Low"].squeeze().astype(float)
        v = df["Volume"].squeeze().astype(float)

        r = float(calc_rsi(c).iloc[-1])
        ml, ms = calc_macd(c)
        e9   = float(calc_ema(c, 9).iloc[-1])
        e20  = float(calc_ema(c, 20).iloc[-1])
        e50  = float(calc_ema(c, 50).iloc[-1])
        e200 = float(calc_ema(c, 200).iloc[-1]) if len(c) >= 200 else float(calc_ema(c, 50).iloc[-1])
        adxv = float(calc_adx(h, l, c).iloc[-1])
        stk  = float(calc_stoch(h, l, c).iloc[-1])
        atrv = float(calc_atr(h, l, c).iloc[-1])
        atr_pct = round(atrv / float(c.iloc[-1]) * 100, 2)
        cl   = float(c.iloc[-1])
        bu, bm, bl = calc_bb(c)
        bw   = float((bu-bl).iloc[-1] / bm.iloc[-1]) if float(bm.iloc[-1]) != 0 else 0
        bwp  = float((bu-bl).iloc[-2] / bm.iloc[-2]) if float(bm.iloc[-2]) != 0 else bw
        vr   = float(v.iloc[-1]) / float(v.rolling(20).mean().iloc[-1])
        stop = round(cl - atrv * 2.0, 2)

        aroon_up, aroon_dn, aroon_osc = calc_aroon(h, l, 14)
        aroon_val    = float(aroon_osc.iloc[-1])
        aroon_up_val = float(aroon_up.iloc[-1])

        obv = calc_obv(c, v)
        obv_trend = float(obv.iloc[-1]) > float(obv.iloc[-6]) if len(obv) >= 6 else False

        chg5 = float((c.iloc[-1] - c.iloc[-6]) / c.iloc[-6] * 100) if len(c) >= 6 else 0

        vol3_trend = (float(v.iloc[-1]) > float(v.iloc[-2])) and (float(v.iloc[-2]) > float(v.iloc[-3])) if len(v) >= 3 else False

        macd_diff = float(ml.iloc[-1] - ms.iloc[-1])

        # ── MOMENTUM YORGUNLUGU TESPITI ──────────────────────────────
        # Gecen hafta analizi: Stoch>73 + 5G>%10 kombinasyonu guvenilmez sinyal
        # Bu kombinasyon "zaten cok yukarida" anlamina geliyor
        momentum_yorgun = (stk > 73) or (chg5 > 10) or (stk > 65 and chg5 > 6)

        # TEKNIK SKOR
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
        # Momentum yorgunlugu cezasi
        if momentum_yorgun: s = max(0, s - 1)

        # HAFTALIK SKOR
        hs = 0
        if 45 <= r <= 65: hs += 2
        # DEG: Stoch ideal penceresi daraltiIdi (eski: <75, yeni: 40-72)
        # Neden: Stoch 73+ olan hisseler gecen hafta tutmadi (KARSN 77.4, VKGYO 78.3)
        if 40 <= stk <= 72: hs += 2      # ideal pencere: ne cok soguk ne cok sicak
        elif stk < 40: hs += 1           # henuz isinmamis, biraz erken ama tamam
        # stk > 72 → puan yok (asiri alim riski)
        if adxv > 25: hs += 2
        if ml.iloc[-1] > ms.iloc[-1]: hs += 2
        if e20 > e50 and e50 > e200: hs += 2
        if aroon_val > 50: hs += 2
        if obv_trend: hs += 2
        if vr > 1.2: hs += 1
        # Momentum yorgunlugu cezasi haftalik skora da uygulanir
        if momentum_yorgun: hs = max(0, hs - 2)

        # GUNLUK SKOR
        gs = 0
        if 50 <= r <= 68: gs += 2
        if adxv > 25: gs += 2
        if aroon_val > 60: gs += 3
        if aroon_up_val > 70: gs += 1
        if vr > 1.5: gs += 3
        if cl > e9: gs += 2
        if atr_pct > 1.5: gs += 2
        if 40 <= stk <= 75: gs += 1
        if ml.iloc[-1] > ms.iloc[-1]: gs += 1

        # PATLAMA SKORU
        ps = 0
        if bw < 0.08: ps += 3
        if 45 <= r <= 58: ps += 2
        if vol3_trend: ps += 3
        if adxv < 25: ps += 2
        if aroon_val > 0: ps += 1
        if macd_diff > -0.5: ps += 1
        if chg5 > -5: ps += 1
        if obv_trend: ps += 2

        return {
            "RSI": round(r, 1), "Stoch": round(stk, 1),
            "MACD": round(macd_diff, 2), "ADX": round(adxv, 1),
            "ATR": round(atrv, 2), "ATR_pct": atr_pct,
            "Stop": stop, "VolR": round(vr, 2),
            "Aroon": round(aroon_val, 1),
            "OBV_trend": obv_trend,
            "EMA9_ok": cl > e9,
            "BB_W": round(bw, 3),
            "Chg5": round(chg5, 1),
            "Vol3_trend": vol3_trend,
            "MomentumYorgun": momentum_yorgun,
            "Skor": s, "HSkor": hs, "GSkor": gs, "PSkor": ps,
            "K1G": karar(r, float(ml.iloc[-1]), float(ms.iloc[-1]))
        }
    except: return None

@st.cache_data(ttl=900)
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
                "Hisse": sembol, "Skor": t["Skor"],
                "HSkor": t["HSkor"], "GSkor": t["GSkor"], "PSkor": t["PSkor"],
                "RSI": t["RSI"], "Stoch": t["Stoch"], "MACD": t["MACD"],
                "ADX": t["ADX"], "ATR_pct": t["ATR_pct"], "Stop": t["Stop"],
                "VolR": t["VolR"], "Aroon": t["Aroon"],
                "BB_W": t["BB_W"], "Chg5": t["Chg5"],
                "OBV": "↑" if t["OBV_trend"] else "↓",
                "EMA9": "✓" if t["EMA9_ok"] else "✗",
                "MomentumYorgun": t["MomentumYorgun"],
                "1G": t["K1G"], "1H": k1h, "1A": k1a,
                "AL_Puan": al_puan
            })
        except: pass
    return pd.DataFrame(sonuc)

def sec_hisseler(df):
    haftalik = df[
        (df["RSI"] >= 45) & (df["RSI"] <= 65) &
        # DEG: Stoch filtresi guncellendi — eski: <75, yeni: <73
        # Neden: Gecen hafta KARSN (77.4) ve VKGYO (78.3) gibi yuksek Stoch'lu
        # hisseler beklenen hareketi yapmadi. 73 ustunde asiri alim riski artiyor.
        (df["Stoch"] < 73) &
        (df["ADX"] > 25) &
        (df["MACD"] > 0) & (df["1G"] == "AL") &
        (df["Chg5"] > -5) &
        # DEG: Momentum yorgunlugu filtresi eklendi
        # Neden: 5G>%10 olan hisseler (CWENE %13.7) zaten cok yukseldi,
        # haftalik listede yer almamali
        (~df["MomentumYorgun"])
    ].copy().sort_values("HSkor", ascending=False).reset_index(drop=True)

    gunluk = df[
        (df["RSI"] >= 50) & (df["RSI"] <= 68) &
        (df["Stoch"] >= 40) & (df["Stoch"] <= 75) &
        (df["ADX"] > 25) & (df["MACD"] > 0) &
        (df["1G"] == "AL") & (df["VolR"] > 1.2) &
        (df["Chg5"] > -5)
    ].copy().sort_values("GSkor", ascending=False).reset_index(drop=True)

    top10 = df[
        (df["RSI"] < 70) &
        # DEG: Top10 Stoch filtresi de guncellendi — eski: <80, yeni: <73
        (df["Stoch"] < 73) &
        (df["Chg5"] > -5)
    ].copy().sort_values(["AL_Puan", "HSkor"], ascending=False).reset_index(drop=True)

    patlama = df[
        (df["BB_W"] < 0.08) &
        (df["RSI"] >= 40) & (df["RSI"] <= 60) &
        (df["ADX"] < 30) &
        (df["Chg5"] > -8)
    ].copy().sort_values("PSkor", ascending=False).reset_index(drop=True)

    return haftalik, gunluk, top10, patlama

def badge(v):
    if v == "AL":
        return "<span style='background:#0d3320;color:#6ee89a;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700'>AL</span>"
    if v == "SAT":
        return "<span style='background:#3d0c0c;color:#f08080;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700'>SAT</span>"
    return "<span style='background:#3d2e00;color:#f0c060;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700'>NOTR</span>"

def skor_html(v, mx=9):
    c = "#1a9e4a" if v >= mx*0.77 else "#e08800" if v >= mx*0.54 else "#c0392b"
    return "<span style='color:" + c + ";font-weight:700'>" + str(v) + "/" + str(mx) + "</span>"

def rsi_html(v):
    c = "#f08080" if v > 70 else "#6ee89a" if v < 30 else "#ccc"
    return "<span style='color:" + c + "'>" + str(v) + "</span>"

def stoch_html(v):
    # DEG: Stoch renklendirmesi guncellendi
    # Yesil: 40-72 (ideal pencere), Turuncu: 73-79 (dikkat), Kirmizi: 80+ (asiri alim)
    if v >= 80:
        return "<span style='color:#f08080;font-weight:700'>" + str(v) + " ⚠</span>"
    elif v >= 73:
        return "<span style='color:#e08800;font-weight:700'>" + str(v) + " ⚠</span>"
    elif 40 <= v <= 72:
        return "<span style='color:#6ee89a'>" + str(v) + "</span>"
    else:
        return "<span style='color:#aaa'>" + str(v) + "</span>"

def chg5_html(v):
    # DEG: 5G% renklendirmesi guncellendi
    # >%10 turuncu uyari: "zaten cok yukseldi, momentum yorgunlugu riski"
    if v > 10:
        return "<span style='color:#e08800;font-weight:700'>" + str(v) + "% ⚠</span>"
    elif v >= 0:
        return "<span style='color:#6ee89a'>" + str(v) + "%</span>"
    else:
        return "<span style='color:#f08080'>" + str(v) + "%</span>"

def tablo_html(df, n, gun_hisse="", hafta_hisseler=None, patlama_hisseler=None):
    if hafta_hisseler is None: hafta_hisseler = []
    if patlama_hisseler is None: patlama_hisseler = []
    satirlar = ""
    for i, r in df.head(n).iterrows():
        is_gun     = r["Hisse"] == gun_hisse
        is_hafta   = r["Hisse"] in hafta_hisseler
        is_patlama = r["Hisse"] in patlama_hisseler
        if is_gun:
            bg = "#2d1f00"; border = "border-left:4px solid #ffd700;"
            label = r["Hisse"] + " ⚡"; color = "#ffd700"
        elif is_hafta:
            bg = "#0d2040"; border = "border-left:4px solid #4da6ff;"
            label = r["Hisse"] + " ★"; color = "#4da6ff"
        elif is_patlama:
            bg = "#1a0d2e"; border = "border-left:4px solid #cc44ff;"
            label = r["Hisse"] + " 🚀"; color = "#cc44ff"
        else:
            bg = "#111"; border = ""; label = r["Hisse"]; color = "#fff"
        mc  = "#6ee89a" if r["MACD"] > 0 else "#f08080"
        ac  = "#6ee89a" if r["ADX"] > 25 else "#aaa"
        arc = "#6ee89a" if r["Aroon"] > 50 else "#aaa"
        obc = "#6ee89a" if r["OBV"] == "↑" else "#f08080"
        e9c = "#6ee89a" if r["EMA9"] == "✓" else "#f08080"
        satirlar += (
            "<tr style='background:" + bg + ";border-bottom:1px solid #1a1a1a;" + border + "'>"
            "<td style='padding:7px 10px;font-weight:700;color:" + color + "'>" + label + "</td>"
            "<td style='text-align:center'>" + skor_html(r["Skor"]) + "</td>"
            "<td style='text-align:center'>" + rsi_html(r["RSI"]) + "</td>"
            "<td style='text-align:center'>" + stoch_html(r["Stoch"]) + "</td>"
            "<td style='text-align:center;color:" + mc + "'>" + str(r["MACD"]) + "</td>"
            "<td style='text-align:center;color:" + ac + "'>" + str(r["ADX"]) + "</td>"
            "<td style='text-align:center;color:" + arc + "'>" + str(r["Aroon"]) + "</td>"
            "<td style='text-align:center;color:" + obc + "'>" + r["OBV"] + "</td>"
            "<td style='text-align:center;color:" + e9c + "'>" + r["EMA9"] + "</td>"
            "<td style='text-align:center;color:#e08800'>" + str(r["ATR_pct"]) + "%</td>"
            "<td style='text-align:center'>" + chg5_html(r["Chg5"]) + "</td>"
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
        "<th style='padding:7px 10px;text-align:left'>Hisse</th>"
        "<th style='padding:7px'>Skor</th><th style='padding:7px'>RSI</th>"
        "<th style='padding:7px'>Stoch</th><th style='padding:7px'>MACD</th>"
        "<th style='padding:7px'>ADX</th><th style='padding:7px'>Aroon</th>"
        "<th style='padding:7px'>OBV</th><th style='padding:7px'>EMA9</th>"
        "<th style='padding:7px'>ATR%</th><th style='padding:7px'>5G%</th>"
        "<th style='padding:7px'>Stop</th>"
        "<th style='padding:7px'>1G</th><th style='padding:7px'>1H</th><th style='padding:7px'>1A</th>"
        "</tr></thead><tbody>" + satirlar + "</tbody></table></div>"
    )

def patlama_tablo_html(df, n=10):
    satirlar = ""
    for i, r in df.head(n).iterrows():
        bb_c = "#cc44ff" if r["BB_W"] < 0.05 else "#e08800"
        mc   = "#6ee89a" if r["MACD"] > 0 else "#f08080"
        obc  = "#6ee89a" if r["OBV"] == "↑" else "#f08080"
        satirlar += (
            "<tr style='background:#1a0d2e;border-bottom:1px solid #2a1a3e;border-left:3px solid #cc44ff'>"
            "<td style='padding:7px 10px;font-weight:700;color:#cc44ff'>" + r["Hisse"] + " 🚀</td>"
            "<td style='text-align:center;color:" + bb_c + ";font-weight:700'>" + str(r["BB_W"]) + "</td>"
            "<td style='text-align:center'>" + rsi_html(r["RSI"]) + "</td>"
            "<td style='text-align:center;color:#ccc'>" + str(r["ADX"]) + "</td>"
            "<td style='text-align:center;color:" + mc + "'>" + str(r["MACD"]) + "</td>"
            "<td style='text-align:center;color:" + obc + "'>" + r["OBV"] + "</td>"
            "<td style='text-align:center'>" + chg5_html(r["Chg5"]) + "</td>"
            "<td style='text-align:center;color:#e08800'>" + str(r["VolR"]) + "x</td>"
            "<td style='text-align:center'>" + badge(r["1G"]) + "</td>"
            "<td style='text-align:center'>" + badge(r["1H"]) + "</td>"
            "<td style='text-align:center'>" + badge(r["1A"]) + "</td>"
            "</tr>"
        )
    return (
        "<div style='overflow-x:auto'>"
        "<table style='width:100%;border-collapse:collapse;font-size:12px;font-family:monospace'>"
        "<thead><tr style='color:#8b949e;border-bottom:2px solid #2a1a3e;background:#0d0a1a'>"
        "<th style='padding:7px 10px;text-align:left'>Hisse</th>"
        "<th style='padding:7px'>BB Genislik</th><th style='padding:7px'>RSI</th>"
        "<th style='padding:7px'>ADX</th><th style='padding:7px'>MACD</th>"
        "<th style='padding:7px'>OBV</th><th style='padding:7px'>5G%</th>"
        "<th style='padding:7px'>Hacim</th>"
        "<th style='padding:7px'>1G</th><th style='padding:7px'>1H</th><th style='padding:7px'>1A</th>"
        "</tr></thead><tbody>" + satirlar + "</tbody></table></div>"
    )

def highlight_box(emoji, baslik, hisse, detay, bg, border_color):
    return (
        "<div style='background:" + bg + ";border-radius:10px;padding:14px 18px;"
        "margin-bottom:8px;border-left:5px solid " + border_color + "'>"
        "<div style='font-size:22px;font-weight:800;color:#fff'>" + emoji + "  " + hisse + "</div>"
        "<div style='font-size:12px;font-weight:600;color:#bbb;margin-top:2px'>" + baslik + "</div>"
        "<div style='font-size:11px;color:#888;margin-top:4px'>" + detay + "</div>"
        "</div>"
    )

def legend_html(tip):
    if tip == "gun":
        items = [
            ("RSI","50-68","Momentum var, kosmamis"),
            ("Stoch","40-75","Hareket odasi var"),
            ("ADX","> 25","Guclu trend"),
            ("Aroon","> 60","Cok taze hareket"),
            ("EMA9","Fiyat ustunde","Kisa vade guclu"),
            ("Hacim","> 1.5x","Yuksek ilgi"),
            ("ATR%","> 1.5%","Gunluk hareket potansiyeli"),
            ("5G%","> -5%","Son 5 gunde cok dusmemis"),
        ]
        baslik = "⚡ Gunluk Kural Seti"; renk = "#ffd700"
    elif tip == "hafta":
        items = [
            ("RSI","45-65","Ne asiri alim ne satim"),
            ("Stoch","40-72","Ideal pencere — 73+ asiri alim riski ⚠"),
            ("ADX","> 25","Trend guclu"),
            ("MACD","Pozitif","Momentum yukari"),
            ("EMA","20>50>200","Tam sirali trend"),
            ("Aroon","> 50","Trend taze"),
            ("OBV","Yukseliyor","Para girisi var"),
            ("5G%","-5% ile +10% arasi","10%+ ise momentum yorgunlugu riski ⚠"),
        ]
        baslik = "★ Haftalik Kural Seti"; renk = "#4da6ff"
    else:
        items = [
            ("BB Genislik","< 0.08","Sikisma var, patlama gelebilir"),
            ("RSI","40-60","Henuz kosmamis"),
            ("ADX","< 30","Yatay ama kirilmak uzere"),
            ("OBV","Yukseliyor","Sessiz para girisi"),
            ("Hacim","Artıyor","Son 3 gun yukseliyor"),
            ("5G%","> -8%","Cok sert dusmemis"),
            ("MACD","Dibe yakin","Donuse hazir"),
        ]
        baslik = "🚀 Patlama Adayi Kural Seti"; renk = "#cc44ff"

    satirlar = ""
    for param, esik, aciklama in items:
        satirlar += (
            "<tr style='border-bottom:1px solid #1a1a1a'>"
            "<td style='padding:5px 8px;color:" + renk + ";font-weight:700;font-size:11px'>" + param + "</td>"
            "<td style='padding:5px 8px;color:#fff;font-size:11px'>" + esik + "</td>"
            "<td style='padding:5px 8px;color:#888;font-size:11px'>" + aciklama + "</td>"
            "</tr>"
        )
    return (
        "<div style='background:#0d1117;border:1px solid #30363d;border-radius:10px;padding:12px;margin-top:10px'>"
        "<div style='color:" + renk + ";font-weight:700;font-size:13px;margin-bottom:8px'>" + baslik + "</div>"
        "<table style='width:100%;border-collapse:collapse'>"
        "<thead><tr style='color:#555'>"
        "<th style='padding:4px 8px;text-align:left;font-size:10px'>Parametre</th>"
        "<th style='padding:4px 8px;text-align:left;font-size:10px'>Esik</th>"
        "<th style='padding:4px 8px;text-align:left;font-size:10px'>Aciklama</th>"
        "</tr></thead><tbody>" + satirlar + "</tbody></table></div>"
    )

# ── SAYFA ─────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center;color:#fff;margin-bottom:4px'>📊 Borsa Dashboard</h1>", unsafe_allow_html=True)
saat = datetime.now().strftime("%d.%m.%Y %H:%M")
st.markdown(
    "<p style='text-align:center;color:#8b949e;font-size:13px'>Guncelleme: " + saat +
    " | ~15 dk gecikmeli | Veri: Yahoo Finance</p>", unsafe_allow_html=True
)

col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
with col_btn2:
    if st.button("🔄 Yenile"):
        st.cache_data.clear()

with st.spinner("BIST100 ve S&P500 taraniyor... (3-5 dakika)"):
    df_bist_raw = tara_tum(BIST100, ".IS")
    df_sp_raw   = tara_tum(SP500, "")

bist_h, bist_g, bist_top, bist_pat = sec_hisseler(df_bist_raw)
sp_h,   sp_g,   sp_top,   sp_pat   = sec_hisseler(df_sp_raw)

bist_gun   = bist_g["Hisse"].iloc[0] if len(bist_g) > 0 else "—"
bist_hafta = bist_h.head(2)["Hisse"].tolist() if len(bist_h) >= 2 else bist_h["Hisse"].tolist()
sp_gun     = sp_g["Hisse"].iloc[0] if len(sp_g) > 0 else "—"
sp_hafta   = sp_h.head(2)["Hisse"].tolist() if len(sp_h) >= 2 else sp_h["Hisse"].tolist()
bist_pat_list = bist_pat.head(5)["Hisse"].tolist() if len(bist_pat) > 0 else []
sp_pat_list   = sp_pat.head(5)["Hisse"].tolist() if len(sp_pat) > 0 else []

st.markdown("---")

# ── HIGHLIGHT KUTULARI ────────────────────────────────────
col_h1, col_h2 = st.columns(2)
with col_h1:
    if bist_gun != "—":
        gr = bist_g.iloc[0]
        st.markdown(highlight_box("⚡", "GUNUN BIST HISSESI — Sabah al, aksam sat", bist_gun,
            "RSI: " + str(gr["RSI"]) + "  |  ADX: " + str(gr["ADX"]) +
            "  |  Aroon: " + str(gr["Aroon"]) + "  |  Hacim: " + str(gr["VolR"]) +
            "x  |  ATR%: " + str(gr["ATR_pct"]) + "%  |  Stop: " + str(gr["Stop"]),
            "#2d1f00", "#ffd700"), unsafe_allow_html=True)
    for idx, h in enumerate(bist_hafta):
        row = bist_h[bist_h["Hisse"] == h].iloc[0]
        st.markdown(highlight_box("★", "HAFTANIN " + str(idx+1) + ". BIST HISSESI — Pzt al, Cuma sat", h,
            "Skor: " + str(row["Skor"]) + "/9  |  RSI: " + str(row["RSI"]) +
            "  |  Aroon: " + str(row["Aroon"]) + "  |  OBV: " + row["OBV"] +
            "  |  1G/1H/1A: " + row["1G"] + "/" + row["1H"] + "/" + row["1A"],
            "#0d2040", "#4da6ff"), unsafe_allow_html=True)

with col_h2:
    if sp_gun != "—":
        gr = sp_g.iloc[0]
        st.markdown(highlight_box("⚡", "GUNUN S&P500 HISSESI — Sabah al, aksam sat", sp_gun,
            "RSI: " + str(gr["RSI"]) + "  |  ADX: " + str(gr["ADX"]) +
            "  |  Aroon: " + str(gr["Aroon"]) + "  |  Hacim: " + str(gr["VolR"]) +
            "x  |  ATR%: " + str(gr["ATR_pct"]) + "%  |  Stop: " + str(gr["Stop"]),
            "#2d1f00", "#ffd700"), unsafe_allow_html=True)
    for idx, h in enumerate(sp_hafta):
        row = sp_h[sp_h["Hisse"] == h].iloc[0]
        st.markdown(highlight_box("★", "HAFTANIN " + str(idx+1) + ". S&P500 HISSESI — Pzt al, Cuma sat", h,
            "Skor: " + str(row["Skor"]) + "/9  |  RSI: " + str(row["RSI"]) +
            "  |  Aroon: " + str(row["Aroon"]) + "  |  OBV: " + row["OBV"] +
            "  |  1G/1H/1A: " + row["1G"] + "/" + row["1H"] + "/" + row["1A"],
            "#0d2040", "#4da6ff"), unsafe_allow_html=True)

st.markdown("---")

# ── PATLAMA ADAYLARI ──────────────────────────────────────
st.markdown("### 🚀 Patlama Adayları — Henüz Koşmamış, Kırılmak Üzere")
st.markdown(
    "<div style='font-size:12px;color:#8b949e;margin-bottom:8px'>"
    "BB sıkışması var + RSI henüz düşük + Hacim artıyor — patlama öncesi sinyali</div>",
    unsafe_allow_html=True
)

col_p1, col_p2 = st.columns(2)
with col_p1:
    st.markdown("#### 🇹🇷 BIST100 — Patlama Adayları")
    if len(bist_pat) > 0:
        st.markdown(patlama_tablo_html(bist_pat, 8), unsafe_allow_html=True)
        st.markdown(legend_html("patlama"), unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:#8b949e'>Şu an kriter karşılayan hisse yok.</p>", unsafe_allow_html=True)

with col_p2:
    st.markdown("#### 🇺🇸 S&P500 — Patlama Adayları")
    if len(sp_pat) > 0:
        st.markdown(patlama_tablo_html(sp_pat, 8), unsafe_allow_html=True)
        st.markdown(legend_html("patlama"), unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:#8b949e'>Şu an kriter karşılayan hisse yok.</p>", unsafe_allow_html=True)

st.markdown("---")

# ── TOP 10 TABLOLAR ───────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🇹🇷 BIST100 — Top 10")
    st.markdown(
        "<div style='font-size:11px;color:#8b949e;margin-bottom:6px'>"
        "⚡ Gunun hissesi | ★ Haftanin hisseleri | 🚀 Patlama adayi | "
        "Filtre: RSI&lt;70, Stoch&lt;73 ⚠, 5G&gt;-5% | ⚠ = Momentum yorgunlugu riski"
        "</div>", unsafe_allow_html=True)
    st.markdown(tablo_html(bist_top, 10, bist_gun, bist_hafta, bist_pat_list), unsafe_allow_html=True)
    with st.expander("📋 Tam liste"):
        st.markdown(tablo_html(bist_top, len(bist_top), bist_gun, bist_hafta, bist_pat_list), unsafe_allow_html=True)
    st.markdown(legend_html("hafta"), unsafe_allow_html=True)
    st.markdown(legend_html("gun"), unsafe_allow_html=True)

with col2:
    st.markdown("### 🇺🇸 S&P500 — Top 10")
    st.markdown(
        "<div style='font-size:11px;color:#8b949e;margin-bottom:6px'>"
        "⚡ Gunun hissesi | ★ Haftanin hisseleri | 🚀 Patlama adayi | "
        "Filtre: RSI&lt;70, Stoch&lt;73 ⚠, 5G&gt;-5% | ⚠ = Momentum yorgunlugu riski"
        "</div>", unsafe_allow_html=True)
    st.markdown(tablo_html(sp_top, 10, sp_gun, sp_hafta, sp_pat_list), unsafe_allow_html=True)
    with st.expander("📋 Tam liste"):
        st.markdown(tablo_html(sp_top, len(sp_top), sp_gun, sp_hafta, sp_pat_list), unsafe_allow_html=True)
    st.markdown(legend_html("hafta"), unsafe_allow_html=True)
    st.markdown(legend_html("gun"), unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#484f58;font-size:11px'>"
    "Teknik analiz amaclidir, yatirim tavsiyesi degildir.</p>",
    unsafe_allow_html=True
)
