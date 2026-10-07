# Financial ecology publication study

## Research focus

This study asks two questions:

1. How are shocks transmitted across connected markets through agent decisions?
2. Do strategies that appear robust in one market remain robust when a linked
   market is introduced?

AML-Sim addresses these questions with controlled interventions rather than a
historical backtest. The same synthetic market can be rerun with an information
link disabled, enabled for observation, or made executable through an
arbitrageur. Paired seeds keep the shock draw and agent randomisation aligned
across those treatments.

## Experimental design

The primary experiment uses a stock and its future. Each run lasts one
simulated hour with 30-second ticks and contains one unexpected AAPL-specific
event at 09:59.

| Treatment | Relationship between markets |
| --- | --- |
| D0 | Markets trade independently; the future receives no stock-event information. |
| I1 | Future participants receive an information-only copy of the stock event. |
| A1 | I1 plus a valuation reference and bounded two-leg basis arbitrage. |

Each treatment is run under ten paired seeds in two decision modes:

- **Frozen:** the configured role strategy remains fixed, providing a
  deterministic benchmark.
- **LLM adaptive:** every trading role retains its fast execution loop while an
  OpenAI-backed slow loop reviews market state every three simulated minutes
  and proposes validated parameter changes.

This produces 60 primary runs: three treatments, two decision modes, and ten
paired seeds. The seed is the independent replication unit. Agents, model
calls, orders, and fills within a run are not treated as independent samples.

A secondary frozen-strategy demonstration adds a bond market and two shocks:
an idiosyncratic stock event and a scheduled policy-rate/liquidity event. Three
paired seeds are run for C0, where the markets share the macro event but have no
stock-future edge, and C2, where that edge and the arbitrageur are enabled. The
six C0/C2 runs demonstrate mechanism coverage; they are not used for a
population-level effect estimate.

## Evidence quality

All 66 accepted runs reached the intended end and contained the expected market,
agent, portfolio, shock, and ecology reports. No accepted run contained a
traceback. The 30 primary LLM runs recorded 4,391 completed slow-loop outcomes,
nine schema rejections that safely retained the previous valid strategy, and no
failed API calls.

Paired treatment effects are reported as seed-level means with 95% percentile
intervals from 10,000 paired bootstrap resamples. Leave-one-seed-out means are
also retained as a sensitivity check. This is stronger than counting orders or
agent observations as independent data, which would overstate the available
evidence.

## Main findings

### Shock transmission

Information visibility alone did not materially change the frozen future
market. Relative to D0, I1 changed event-window future volume by 0.0 shares
(95% CI [-0.3, 0.3]) and trade count by 0.0. The relationship graph therefore
delivered the event as designed, but deterministic future agents did not trade
on that information strongly enough to transmit price pressure.

Making the relationship executable changed the result. Relative to I1, frozen
A1 increased event-window future volume by 44.5 shares (95% CI [36.5, 52.5])
and trade count by 4.5 (95% CI [3.8, 5.2]); both effects had the same direction
in all ten seeds. The future event return changed by -0.062 percentage points
(95% CI [-0.135, -0.002]). The channel ledger connects these market-level
effects to relationship observations, arbitrage decisions, child orders, and
fills.

The appropriate conclusion is therefore mechanism-specific: a declared
information edge changes who can observe a shock, but transmission becomes
economically visible when participants have both an incentive and an
executable route to trade across the relationship.

### Strategy robustness

Frozen role strategies remained comparatively stable across D0, I1, and A1.
Adaptive responses were more sensitive to the linked treatment. In A1 versus
D0, LLM market makers had a mean ROI change of -80.0 basis points (95% CI
[-148.4, -9.4]) and institutional traders changed by -4.2 basis points (95% CI
[-6.6, -1.7]); the retail interval crossed zero.

These values are not interpreted as proof that LLMs are inferior traders. The
adaptive primary runs also produced 977 self-cross-prevention events, compared
with 81 in frozen runs. Strategy adaptation and execution quality therefore
changed together. The defensible finding is that robustness did not transfer
uniformly to the linked market and that market-making was the most sensitive
role under the current LLM control contract.

### Secondary three-market demonstration

C0/C2 separates two kinds of exposure. The local stock event reaches the future
only through the enabled relationship, while the scheduled macro event reaches
the stock, future, and bond directly through shared market state. C2 recorded
25 fully hedged, 13 partial or unhedged, and one unfilled cross-market decision.
This verifies the stock-future-bond architecture and its audit trail, but the
three paired seeds support descriptive mechanism evidence only.

## Reproduce the analysis

The analysis expects accepted runs named
`pub1_ecology_<treatment>_<mode>_r<seed>` under `.aml_runs/`, for example
`pub1_ecology_a1_llm_r04`.

```bash
source .venv/bin/activate
python -m pip install -r analysis/requirements.txt
python analysis/financial_ecology_publication_analysis.py
```

By default, tables, figures, run-quality checks, a claim manifest, and the
written findings are generated under
`artifacts/financial_ecology_publication_v1/analysis/`. Raw runs and generated
artifacts are intentionally ignored by Git; the analysis code and experimental
contract are versioned here.

## Interpretation boundary

The estimates describe the configured synthetic ecology. They are not forecasts
for AAPL futures or Treasury markets, and the scenarios are not calibrated to a
particular trading day. The primary result supports causal statements about the
implemented information and execution channels. The secondary C0/C2 result
supports architectural claims, not a general estimate of macro-shock
transmission.
