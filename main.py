"""gta6-trophy-rarity-forecast-01 entry: a calibrated forecaster for Exophase earn rates.

Run by the grader as `python3 /bot/main.py` (Python 3.13, standard library only, no
network). Reads every *.in case in $MAKEFUN_INPUT_DIR (else /input) and writes one
<id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output): five quantiles (0.05, 0.25,
0.5, 0.75, 0.95) of each trophy's earn rate, in percent, at the case's target.

Approach (see APPROACH.md for the long version):

  * Exophase refreshes the page on its own schedule, so consecutive house reads often
    repeat one stale snapshot. Only *distinct* reads (players tracked or any rate
    changed) carry new information; the fit uses those.
  * The first distinct segment of a case can be a pre-launch plateau (a tiny early
    cohort, replaced wholesale at the first refresh). Points before the first change
    in players tracked belong to that dead regime and are dropped.
  * Per trophy, the logit of the earn rate is fit against ln(1 + hours since release)
    by recency-weighted least squares (36 h half-life). Log-time is the natural clock
    here: progression and adoption both decelerate, so a line in log-time extrapolates
    the 4.5-day horizon as a mild step beyond the observed range instead of a leap.
  * Slopes are shrunk toward their tier's pooled slope (platinum and gold, with too
    few members, shrink toward the field), with shrinkage strength from each fit's
    own slope standard error: a trophy with a clean, long trend keeps it; a noisy or
    quantized one borrows the tier's behaviour. A series pinned at the 2-decimal
    print floor (a platinum at 0.01%) is censored, not precise, so its error scale
    gets a floor from the print quantum and it shrinks like a noisy series. The
    intercept is refit through the recent cloud, and the implied move from the
    recent level is damped by the headroom under a week-one ceiling and passed
    through a soft clamp, so no extrapolation runs away.
  * The spread is honest, not decorative: the house random-walk scale
    0.5 * sqrt(horizon days) combined with the fit's residual scale grown by the
    extrapolation distance, plus an inflation when the page has been stale for long
    before the checkpoint (a refresh is pending and its jump is unobserved). A flat
    history (no distinct reads at all, as in the launch-night practice cases) falls
    back to the last rate with a wide spread and, in the launch window, a small
    downward dilution shift: at a launch refresh the tracked base multiplies faster
    than earners accumulate, so rates fall.
  * Robustness first: every per-trophy and per-case failure degrades to a valid
    fallback row (last rate, wide spread) rather than an invalid case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [NormalDist().inv_cdf(t) for t in LEVELS]

# Model constants (chosen a priori, not tuned on the practice answers).
RECENCY_HALF_LIFE_H = 36.0    # weight half-life of a read, in hours
WALK_SIGMA = 0.5              # random-walk spread per sqrt-day, as the house baseline
FLAT_SIGMA = 1.2              # spread when the history holds no distinct read at all
STALE_SIGMA = 0.55            # extra spread (in quadrature) when a refresh is overdue
SOFT_MOVE = 2.2               # soft clamp scale on the median's logit move
LAUNCH_DILUTION = -0.35       # logit shift for a flat launch-window history
SLOPE_PRIOR_VAR = 0.09        # variance of true slopes around their group mean
CEILING = 97.0                # a week-one rate's practical ceiling, for headroom damping


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y):
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints (JSON-compact)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_between(a, b):
    return (b - a).total_seconds() / 3600.0


def release_anchor(case):
    """Midnight UTC on the game's release date; falls back to the first read."""
    try:
        d = parse_time(case["game"]["release_date"] + "T00:00:00Z")
        return d.astimezone(timezone.utc)
    except Exception:
        try:
            return parse_time(case["history"][0]["at"])
        except Exception:
            return datetime(2026, 11, 19, tzinfo=timezone.utc)


def distinct_reads(case, anchor):
    """Reads that carry new information: players tracked or any rate changed.

    Returns a list of (hours_since_release, players_tracked, rates) oldest first.
    """
    out = []
    prev_key = None
    for read in case["history"]:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        try:
            t = parse_time(read.get("at") or read.get("slot"))
        except Exception:
            continue
        out.append((hours_between(anchor, t), read.get("players_tracked"), rates))
    return out


def last_rate(history, idx):
    """A trophy's rate in the newest read that has one, else None."""
    for read in reversed(history):
        rates = read.get("rates") or []
        if idx < len(rates) and rates[idx] is not None:
            return rates[idx]
    return None


def wls(points):
    """Weighted least squares y ~ a + b*x over (x, y, w) points.

    Returns (a, b, resid_sd, slope_se, sxx) with weights normalised to sum 1.
    """
    sw = sum(w for _, _, w in points)
    if sw <= 0:
        return 0.0, 0.0, 1.0, float("inf"), 0.0
    pts = [(x, y, w / sw) for x, y, w in points]
    mx = sum(w * x for x, _, w in pts)
    my = sum(w * y for _, y, w in pts)
    sxx = sum(w * (x - mx) ** 2 for x, _, w in pts)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in pts)
    if sxx < 1e-9:
        b = 0.0
    else:
        b = sxy / sxx
    a = my - b * mx
    resid = [(y - (a + b * x), w) for x, y, w in pts]
    var = sum(w * e * e for e, w in resid)
    n = len(pts)
    if n > 3:
        var *= n / (n - 2.0)
    sd = math.sqrt(max(var, 1e-6))
    se = sd / math.sqrt(sxx) if sxx >= 1e-9 else float("inf")
    return a, b, sd, se, sxx


def quantile_row(mu, sigma):
    vals = []
    prev = 0.0
    for z in Z:
        v = round4(inv_logit(mu + z * sigma))
        v = min(100.0, max(0.0, float(v)))
        if v < prev:
            v = prev
        vals.append(v)
        prev = v
    return vals


def fallback_forecast(case, reads, horizon_days):
    """Last-rate rows with a wide spread; the launch-window dilution shift applies
    when the page never refreshed inside the history (a stale launch snapshot)."""
    flat = len(reads) <= 1
    sigma = WALK_SIGMA * math.sqrt(horizon_days)
    if flat:
        sigma = max(sigma, FLAT_SIGMA)
    shift = 0.0
    if flat and reads:
        h_last, players_last, _ = reads[-1]
        if h_last <= 12.0 and (players_last or 0) <= 20000:
            shift = LAUNCH_DILUTION
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r = last_rate(case["history"], i)
        mu = logit(r if r is not None else 0.0) + shift
        rows.append({"id": trophy["id"], "q": quantile_row(mu, sigma)})
    return {"trophies": rows}


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    anchor = release_anchor(case)
    horizon_days = float(case["horizon_minutes"]) / 1440.0
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    x_target = math.log1p(max(0.0, hours_between(anchor, target)))

    reads = distinct_reads(case, anchor)
    # Drop the dead regime: points before players tracked first changed.
    if len(reads) >= 3:
        first_players = reads[0][1]
        cut = 0
        for j, (_, p, _) in enumerate(reads):
            if p != first_players:
                cut = j
                break
        if 0 < cut < len(reads) - 2:
            reads = reads[cut:]
    if len(reads) < 3:
        return fallback_forecast(case, reads, horizon_days)

    h_last = reads[-1][0]
    x_last = math.log1p(max(0.0, h_last))

    # Staleness: how long the page had been unchanged at the checkpoint, against
    # the typical gap between refreshes seen in this case.
    gaps = [reads[j + 1][0] - reads[j][0] for j in range(len(reads) - 1)]
    gaps = sorted(g for g in gaps if g > 0)
    typical_gap = gaps[len(gaps) // 2] if gaps else None
    stale_h = hours_between(anchor, checkpoint) - h_last
    stale = stale_h > 8.0 and (typical_gap is None or stale_h > 2.5 * typical_gap)

    # Per-trophy trend fits.
    fits = {}
    for i, trophy in enumerate(trophies):
        pts = []
        rates_i = []
        for h, _, rates in reads:
            if i < len(rates) and rates[i] is not None:
                x = math.log1p(max(0.0, h))
                w = 0.5 ** (max(0.0, h_last - h) / RECENCY_HALF_LIFE_H)
                pts.append((x, logit(rates[i]), w))
                rates_i.append(rates[i])
        if len(pts) >= 3:
            a, b, sd, se, sxx = wls(pts)
            if sxx >= 1e-6:
                # Print-floor honesty: rates are printed to 2 decimals, so a trophy
                # sitting at 0.01-0.05% shows a perfectly flat series that is really
                # a censored one. Its residual scale and slope error get a floor from
                # the print quantum, so the fit is not mistaken for precise.
                med_rate = sorted(rates_i)[len(rates_i) // 2] if rates_i else 0.0
                quant_sd = min(1.0, 0.30 * 0.01 / max(med_rate, 0.01))
                sd = max(sd, quant_sd)
                se = sd / math.sqrt(sxx)
                fits[i] = {"pts": pts, "a": a, "b": b, "sd": sd, "se": se}
                continue
        fits[i] = None

    # Group slopes: tier where the tier has company, else the whole field.
    tier_counts = {}
    for t in trophies:
        tier_counts[t["tier"]] = tier_counts.get(t["tier"], 0) + 1

    def pooled(indices):
        num = 0.0
        den = 0.0
        for i in indices:
            f = fits.get(i)
            if f is None:
                continue
            p = 1.0 / (f["se"] ** 2 + 0.04) if math.isfinite(f["se"]) else 0.0
            num += p * f["b"]
            den += p
        return num / den if den > 0 else 0.0

    all_idx = list(range(len(trophies)))
    b_all = pooled(all_idx)
    group_slope = {}
    for tier in tier_counts:
        if tier_counts[tier] >= 3:
            idx = [i for i, t in enumerate(trophies) if t["tier"] == tier]
            group_slope[tier] = 0.5 * pooled(idx) + 0.5 * b_all
        else:
            group_slope[tier] = b_all

    base_sigma = WALK_SIGMA * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(trophies):
        try:
            rows.append(trend_row(i, trophy, fits.get(i), group_slope[trophy["tier"]],
                                  reads, x_target, x_last, h_last, base_sigma, stale, case))
        except Exception:
            r = last_rate(history, i)
            mu = logit(r if r is not None else 0.0)
            rows.append({"id": trophy["id"], "q": quantile_row(mu, max(base_sigma, 0.6))})
    return {"trophies": rows}


def damped_move(move, level):
    """Tame an extrapolated logit move. A positive move is damped by the headroom
    left under the week-one ceiling: a trophy already at 85% has nearly exhausted
    its earning curve, so a riser's slope cannot carry it to 97% in four days.
    Every move then passes a soft clamp, so no fitted line runs away."""
    if move > 0:
        head = (logit(CEILING) - level) / logit(CEILING)
        move *= min(1.0, max(0.30, head))
    return SOFT_MOVE * math.tanh(move / SOFT_MOVE)


def trend_row(i, trophy, fit, b_group, reads, x_target, x_last, h_last,
              base_sigma, stale, case):
    r_last = last_rate(case["history"], i)
    y_last = logit(r_last if r_last is not None else 0.0)
    if fit is None:
        # No usable trend of its own: ride the group's slope from the last level.
        mu = y_last + damped_move(b_group * (x_target - x_last), y_last)
        sigma = max(base_sigma, 0.75)
        if stale:
            sigma = math.hypot(sigma, STALE_SIGMA)
        return {"id": trophy["id"], "q": quantile_row(mu, min(2.3, sigma))}

    pts = fit["pts"]
    # Shrink the slope toward the group's by precision.
    p_i = 1.0 / (fit["se"] ** 2 + 0.04) if math.isfinite(fit["se"]) else 0.0
    p_g = 1.0 / SLOPE_PRIOR_VAR
    b = (p_i * fit["b"] + p_g * b_group) / (p_i + p_g) if (p_i + p_g) > 0 else b_group
    # Refit the intercept through the cloud with the shrunk slope fixed.
    sw = sum(w for _, _, w in pts)
    a = sum(w * (y - b * x) for x, y, w in pts) / sw
    # Recent level: the fit's own level at the last read, blended with the raw one.
    recent = a + b * x_last
    level = 0.5 * recent + 0.5 * y_last
    mu = level + damped_move(a + b * x_target - level, level)

    sd = max(fit["sd"], 0.06)
    extrap = max(0.0, x_target - x_last)
    sigma_fit = min(sd, 1.2) * (1.0 + 0.6 * extrap)
    sigma = math.hypot(base_sigma, sigma_fit)
    if stale:
        sigma = math.hypot(sigma, STALE_SIGMA)
    sigma = min(2.3, max(0.30, sigma))
    return {"id": trophy["id"], "q": quantile_row(mu, sigma)}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            forecast = forecast_case(case)
        except Exception:
            # Last-resort: rebuild a valid forecast from the raw history alone.
            try:
                anchor = release_anchor(case)
                reads = distinct_reads(case, anchor)
                forecast = fallback_forecast(
                    case, reads, float(case["horizon_minutes"]) / 1440.0)
            except Exception:
                continue
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
