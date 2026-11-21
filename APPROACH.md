# Approach

**What is being forecast.** Each rate is earners ÷ players tracked, printed to 2
decimals. Two forces move it: the cohort keeps earning (rates rise), and new
players keep arriving with nothing earned (rates dilute). On launch night the
second force won in one step: the tracked base jumped 638 → 3,868 at the first
refresh and nearly every rate fell. Over the final's 4.5 days both forces are at
work; the net differs by trophy, which is why the model fits each trophy's own
trajectory instead of applying one global drift.

**Reads.** Exophase refreshes on its own schedule, so most house reads repeat a
stale snapshot. Only distinct reads (players or any rate changed) enter the fit,
and the opening segment before players tracked first changes — the pre-launch
plateau — is dropped as a dead regime.

**Median.** Per trophy, logit(rate) is fit against ln(1 + hours since release)
by recency-weighted least squares (36 h half-life). Log-time is the right clock:
adoption and progression both decelerate, so the line's step from the checkpoint
(x ≈ 4.1) to the target (x ≈ 5.1) is a mild extrapolation. Slopes are shrunk
toward the tier's pooled slope (platinum/gold shrink toward the field) with
strength from each fit's slope standard error. A series pinned at the print
floor (0.01–0.05%) is censored, not precise: its error scale gets a floor from
the print quantum, so it shrinks like the noisy series it is. The extrapolated
move is damped by the headroom under a 97% week-one ceiling (an 85% story
trophy cannot keep a riser's slope) and passed through a soft clamp.

**Spread.** σ combines the house random-walk scale 0.5·√(horizon days) with the
fit's residual scale grown by extrapolation distance, inflated when the page
had been stale long before the checkpoint (a refresh — and its jump — is still
pending), and floored by the print-quantum term for floor-pinned trophies.

**Fallbacks.** A history with no distinct reads (the launch-night practice
shape) yields the last rate with a wide σ = 1.2 spread and, inside the launch
window, a −0.35 logit dilution shift. Any per-trophy or per-case failure
degrades to a valid last-rate row rather than an invalid case.

**Checks.** The program is deterministic (same input → same bytes), runs a
55-trophy case in well under a second locally, and was validated against the
output spec on both practice cases plus synthetic final-shaped cases with
cohort-based ground truth, where it scored roughly 40% below the house
baseline's pinball loss.
