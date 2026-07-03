# Data sources and curation status

**Status: ALL VALUES ARE PLACEHOLDERS.** Every record carries a `source` field;
none has yet been curated against published tables. The schemas are final; the
numbers are transparent, debatable defaults in the right ballpark.

| File | Field | Placeholder basis | Curation target |
|---|---|---|---|
| `tasks.json` | `exposure` (E0/E1/E2) | Hand-assigned following the class definitions in Eloundou et al. (2023), "GPTs are GPTs" | Map to their published occupation/task exposure tables |
| `tasks.json` | `difficulty` | Hand-assigned; E2 < E1 < E0 ordering | Calibrate to capability-benchmark progress + Anthropic Economic Index task usage |
| `tasks.json` | `augmentation` | Hand-assigned, 0–0.45 | Estimates from AI-productivity RCTs (e.g. support agents, coding studies) |
| `occupations.json` | `employment_share`, `base_wage` | Approximate BLS OES magnitudes for SOC major groups | BLS OES May tables (latest release) |
| `occupations.json` | `task_weights` | Hand-assigned sparse vectors | O*NET generalized work activity importance ratings |
| `sectors.json` | `employment_share`, `demand_elasticity` | Carried over from the legacy aggregate model (`legacy/model.js`) | National accounts / literature elasticities |
| `sectors.json` | `occupation_mix` | Hand-assigned | BLS industry–occupation matrix |
| `sectors.json` | firm size params | Truncated Pareto, alpha 1.3–1.6 | Business dynamics statistics firm-size distributions |

## References

- Eloundou, Manning, Mishkin, Rock (2023): GPTs are GPTs — occupational
  exposure classes E0/E1/E2.
- Anthropic Economic Index — observed task-level AI usage shares.
- Korinek (NBER w32980); Korinek & Suh (NBER w32255) — capability vs. cost
  scenario axes.
- Hampole, Papanikolaou, Schmidt, Seegmiller (NBER w33509) — AI and labour
  demand evidence.
