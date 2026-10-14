"""
diagnose.py — 把用户指标对照德约 IQR band 生成诊断报告
=====================================================
连续评分(非二值): 每项按"与德约中位数的距离/区间宽度"算 0–100 子分,
总分=各项均值。并把"发力顺序"(各环节峰值时刻)摊开展示, 去黑箱。
"""
from __future__ import annotations

# direction: "higher"=越大越好(到中位即满分), "band"=越接近德约区间越好。
METRIC_META = {
    "seq_lead": {
        "label": "發力鏈順序（由近端到遠端）", "unit": "", "direction": "band",
        "good": "先由下半身與軀幹帶動，再依序將力量傳到手臂，動力鏈流暢，與職業球員的動作相近。",
        "below": "近端帶動不足（各部位幾乎同時發力），容易變成「只用手臂打球」。引拍後先轉髖、帶動髖部，再讓手臂稍晚依序跟上並加速揮出。",
        "above": "近端帶動過多，建議留意各環節之間的銜接，不讓動作脫節。",
    },
    "xfactor_magnitude": {
        "label": "X-factor 身體蓄力幅度", "unit": "°", "direction": "band",
        "good": "上下半身適度分離，蓄力到位。",
        "above": "肩髖分離角度偏大，注意避免過度扭轉，以免回正速度變慢或增加腰部負擔。",
        "below": "上下半身分離不足，蓄力幅度較小。引拍時可多轉肩、穩住下盤，增加肩髖夾角以累積力量。",
    },
    "contact_forward": {
        "label": "擊球點・前後位置", "unit": "", "direction": "band",
        "good": "擊球點相對身體的位置與職業球員相近，能充分運用身體向前帶動的力量。",
        "below": "擊球點偏後（離身體太近），容易被來球擠壓、難以順利發力。可嘗試讓擊球點更靠前，提早揮拍。",
        "above": "擊球點太靠前，可能不容易扎實擊球、發力不穩，請留意觸球時機。",
    },
    "contact_height": {
        "label": "擊球點・高度", "unit": "", "direction": "band",
        "good": "擊球高度與職業球員相近，位於較容易發力的區間。",
        "below": "擊球點偏低，常出現在腰部以下；可留意降低重心、提早準備，或選擇較高的擊球點。",
        "above": "擊球點偏高，請留意是否被高球影響揮拍舒適度、上旋與穩定性。",
    },
    "rot_pre_frac": {
        "label": "擊球前的身體旋轉完成度", "unit": "", "direction": "higher",
        "good": "擊球前髖部與肩膀已轉到位，觸球時將旋轉累積的力量傳給手臂。",
        "below": "身體轉體偏晚——髖部與肩膀的大部分旋轉都發生在擊球之後（轉過頭了）。"
                 "練習建議：引拍後先轉髖再轉肩、並在觸球前完成轉體；觸球時身體已朝向來球方向，接著將力量傳給手臂揮出。",
        "above": "",
    },
}
ORDER = ["seq_lead", "xfactor_magnitude", "contact_forward", "contact_height", "rot_pre_frac"]
SEQ_LABELS = {"hip": "髖部", "shoulder": "肩膀", "upper_arm": "上臂",
              "forearm": "前臂", "wrist": "手腕"}
SEQ_ORDER = ["hip", "shoulder", "upper_arm", "forearm", "wrist"]


# 时序类指标可靠所需的最少装载帧数(引拍底→击球)。30fps 常速业余挥拍通常只有 3~6 帧,
# 量化噪声≈带宽级(正控实验: 同一挥拍两次测 seq_lead 差 0.2+), 低于此阈值锁定不评。
TIMING_MIN_LOADING = 10
# rot_pre 也锁: 短装载下窗口相位失真, 连德约本人都只测出 0.14(正控回归实测)——
# 它反映的是拍摄条件而非技术, 30fps 常速下如实只评击球点两项。
TIMING_LOCKED = ("seq_lead", "xfactor_magnitude", "rot_pre_frac")
LOCK_TIP = ("這項指標需要足夠的時間解析度；你的前揮只有幾格，無法在物理上可靠地量測精細時序。"
            "使用手機慢動作模式（120/240 fps）拍攝即可解鎖。")


def _acceptable(band: dict):
    """合格区 [P10,P90]。旧版参考带无 p10/p90 时按近正态从四分位外推(×1.9)。"""
    q1, q3, med = band["lo"], band["hi"], band["median"]
    p10 = band.get("p10", med - 1.9 * max(med - q1, 1e-6))
    p90 = band.get("p90", med + 1.9 * max(q3 - med, 1e-6))
    return p10, p90


def _subscore(val: float, band: dict, direction: str) -> float:
    """0–100 连续子分。IQR(中间50%)内=100; IQR边→P10/P90 线性 100→75;
    合格区外再一个半合格区宽度线性衰减到 0。
    设计依据: IQR 当合格线会把一半职业挥拍判"出带"(正控实验), 合格区必须是 P10-P90。"""
    q1, q3, med = band["lo"], band["hi"], band["median"]
    if direction == "higher" and val >= med:
        return 100.0
    if direction == "lower" and val <= med:
        return 100.0
    if q1 <= val <= q3:
        return 100.0
    p10, p90 = _acceptable(band)
    edge, outer = (q1, p10) if val < q1 else (q3, p90)
    span = max(abs(outer - edge), 1e-6)
    d = abs(val - edge)
    if d <= span:
        return round(100.0 - 25.0 * d / span, 1)
    tail = max((p90 - p10) / 2.0, 1e-6)
    return round(max(0.0, 75.0 - 75.0 * (d - span) / tail), 1)


def _status(val: float, band: dict) -> str:
    if band["lo"] <= val <= band["hi"]:
        return "in_band"
    p10, p90 = _acceptable(band)
    if p10 <= val <= p90:
        return "acceptable"
    return "below" if val < band["lo"] else "above"


def diagnose(user_metrics: dict, reference: dict,
             loading_frames: int = None, n_swings: int = 1) -> dict:
    """loading_frames: 装载帧数(时序指标置信门, None=不启用锁);
    n_swings: 聚合的挥拍数(>=2 时说明评分基于多挥拍中位数)。"""
    band_all = reference["metrics_band"]
    timing_locked = loading_frames is not None and loading_frames < TIMING_MIN_LOADING
    items, subs, weights = [], [], []
    locked_keys = []
    for key in ORDER:
        meta = METRIC_META[key]
        band = band_all[key]
        val = float(user_metrics[key])
        locked = timing_locked and key in TIMING_LOCKED
        p10, p90 = _acceptable(band)
        if locked:
            locked_keys.append(key)
            items.append({
                "key": key, "label": meta["label"], "unit": meta["unit"],
                "value": round(val, 2), "band_lo": round(band["lo"], 2),
                "band_hi": round(band["hi"], 2), "median": round(band["median"], 2),
                "p10": round(p10, 2), "p90": round(p90, 2),
                "score": None, "status": "low_confidence", "ok": None,
                "locked": True, "tip": LOCK_TIP,
            })
            continue
        sub = _subscore(val, band, meta["direction"])
        subs.append(sub)
        weights.append(1.0)
        status = _status(val, band)
        better_side = (meta["direction"] == "higher" and val > band["hi"]) or \
                      (meta["direction"] == "lower" and val < band["lo"])
        if status == "in_band" or better_side:
            tip, ok = meta["good"], True
        elif status == "acceptable":
            tip, ok = meta["good"] + "（落在職業球員常見的波動範圍內）", True
        elif status == "above":
            tip, ok = meta["above"], False
        else:
            tip, ok = meta["below"], False
        items.append({
            "key": key, "label": meta["label"], "unit": meta["unit"],
            "value": round(val, 2), "band_lo": round(band["lo"], 2),
            "band_hi": round(band["hi"], 2), "median": round(band["median"], 2),
            "p10": round(p10, 2), "p90": round(p90, 2),
            "score": sub, "status": status, "ok": ok,
            "locked": False, "tip": tip,
        })

    score = round(sum(subs) / max(sum(weights), 1e-6))
    if score >= 80:
        summary = "整體動力鏈接近職業水準，請繼續保持。"
    elif score >= 60:
        summary = "動力鏈基礎不錯，有 1～2 個環節可以優先加強。"
    else:
        summary = "動力結構還有明顯的改善空間，建議先從下半身帶動的練習開始。"
    prefix = f"基於 {n_swings} 次揮拍的綜合評估。" if n_swings >= 2 else ""
    suffix = (f"（這次可評 {len(subs)}/{len(ORDER)} 項，時序類需慢動作影片才能解鎖）"
              if locked_keys else "")
    summary = prefix + summary + suffix

    # 发力顺序透明展示 (各环节峰值相对击球的时刻)
    pt = user_metrics.get("peak_times") or {}
    rpt = (reference.get("reference", {}) or {}).get("peak_times") or {}
    sequencing = [{"seg": s, "label": SEQ_LABELS[s],
                   "t": pt.get(s), "ref_t": rpt.get(s)}
                  for s in SEQ_ORDER if pt.get(s) is not None]

    return {"score": score, "summary": summary, "items": items,
            "sequencing": sequencing, "n_swings": n_swings,
            "locked": locked_keys}
