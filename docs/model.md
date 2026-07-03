# Model description

An agent-based model of AI adoption at the organizational and labour-market
level. ~5,000 workers and ~400–500 firms interact on a monthly clock; S-curves,
adoption cascades, displacement waves, wage scarring, and Beveridge-curve
shifts **emerge** from micro-interactions. There is no sigmoid over time
anywhere in the code.

## Entities

- **Tasks** (`labour_sim/data/tasks.json`): ~24 generalized work activities with
  Eloundou-et-al.-style exposure classes (E0 not exposed / E1 LLM-exposed /
  E2 LLM+software), a `difficulty` (capability level at which AI clears the
  task), and an `augmentation` boost for humans on non-automated exposed tasks.
- **Occupations** (22 SOC major groups): task-weight vectors, base wages,
  employment shares. Occupation distance = 1 − cosine(task weights); retraining
  feasibility therefore *derives from the task data*.
- **Workers**: occupation, lognormal skill, state (employed / searching /
  discouraged), wage, reservation wage, unemployment spell, tenure.
- **Firms**: sector, per-occupation headcount targets, vacancies with
  escalating offers, per-task adoption set, heterogeneous adoption hurdle,
  price, demand factor, loss streak, age.

## Tick order (monthly)

1. **Capability** (`capability.py`) — logistic frontier toward a ceiling with
   seeded AR(1) shocks; per-task quality `q = clamp((c_eff − difficulty)/(1 −
   difficulty))`; E1 tasks see a scaffolding penalty. AI price declines
   exponentially — capability and cost are separate levers (Korinek/Suh).
2. **Adoption** (`adoption.py`) — each firm re-evaluates a few sampled tasks:
   ROI = (mean wage − AI cost/q − amortised adjustment cost)/wage; adopt with
   probability logistic(ROI + imitation·sector peer share − firm hurdle).
   Adoption is absorbing.
3. **Product market** (`demand.py`) — automation and augmentation cut unit
   costs; 60% pass-through to prices; iso-elastic sector demand expands
   (endogenous rebound); size-weighted logit price competition splits a
   conserved sector volume (entrants dilute incumbents, never add net demand).
4. **Labour demand** (`labour_demand.py`) — per-occupation targets =
   base × demand factor × (1 − automated share)/(1 + augmentation); firing
   costs shed surplus gradually (lowest tenure first).
5. **Separations** (`labour_flows.py`) — baseline quits at ~1.2%/month.
6. **Vacancies** — firms post deficits; unfilled offers escalate 2%/tick.
7. **Matching** (`matching.py`) — searchers send ~5 applications weighted by
   task-distance and offer-vs-reservation; short spells search near their
   occupation, long spells anywhere (retraining channel, with a skill discount
   softened by the retraining subsidy); firms rank by effective skill; a
   screening-efficiency coin gates completion. Wage = β·MP + (1−β)·reservation,
   floored at the benefit level.
8. **Unemployed updates** — reservation decays 1%/tick to the benefit floor;
   24-month spells discourage; discouraged re-enter at 3%/tick.
9. **Employed wages** (`wages.py`) — drift 3%/tick toward marginal product,
   which falls with automation of the worker's tasks and rises with
   augmentation → wage scarring and inequality emerge.
10. **Entry/exit** (`entry_exit.py`) — sustained demand shortfall → exit;
    replacement entry keeps the stationary baseline stationary; sector booms
    attract extra entrants that start with lower adoption hurdles (vintage
    effect — an accelerant of tipping).

## Calibration anchors (locked as tests)

The no-AI baseline must hold: unemployment ~3–8%, job-finding ~10–45%/month,
separations ~0.6–2.5%/month, wage Gini 0.20–0.45, no employment drift. See
`tests/test_baseline_calibration.py` — any new mechanism must keep these green.

## Emergent findings already visible

- **S-curves without sigmoids**: adoption builds slowly, accelerates after
  capability crosses the task-difficulty mass and imitation kicks in
  (`tests/test_ai_integration.py::test_s_curve_emerges_not_instant`).
- **Displacement with partial rebound**: fast takeoff spikes unemployment to
  ~15–25% while output rises ~80% — productivity and distress coexist.
- **Policy levers are seed- and specification-sensitive** (5-seed medians of
  peak distress under fast takeoff, `paper/figures/_policy.json`): benefit
  generosity cushions in the median (0.293 → 0.228) but worsens 2/5 seeds, and
  its sign flipped across development revisions of the demand block; firing
  costs are a wash in the median (0.240 vs 0.234) with seed-level sign flips.
  Report multi-seed medians with dispersion, never single runs.
- **Only matching efficiency is mechanically signed**: less reallocation
  friction → weakly lower peak distress in the median (0.266 → 0.226; locked
  as a gate test).
- **Wage collapse with compression**: under fast takeoff the median wage
  roughly halves while within-labour Gini slightly *falls* — the distribution
  compresses toward the benefit floor as surplus shifts from labour to firms.

## Limitations

- Data values are placeholders (see `docs/data-sources.md`).
- No robotics channel: E0 (physical) tasks never automate.
- No aggregate demand feedback from unemployment to consumption.
- Discouragement/re-entry rates are stylized, not estimated.
