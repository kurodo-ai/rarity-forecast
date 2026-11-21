# rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kurodo-ai/rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kurodo-ai/rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A calibrated forecaster for GTA VI's Exophase trophy earn rates. Because Exophase refreshes the page on its own schedule, the model fits only the distinct reads in each case, extrapolating every trophy's logit rate against log-time since release with slopes shrunk toward its tier's pooled behaviour, moves damped by the headroom under a week-one ceiling, and a spread that combines the random-walk scale with each fit's own residual scale — wider when a series is pinned at the print floor or the page has gone stale. Degenerate histories fall back to the last rate with a wide, dilution-aware spread, so every case produces a valid forecast.

## Run it

```sh
python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=gzzxjh8r&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
