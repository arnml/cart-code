# Cross-Model Analysis: CART Targets Overlap

**Generated:** 2026-03-28 13:00:01

This analysis identifies which CART targets are consistent across models,
helping prioritize the most important questions for CART to solve.

---

## Summary

| Category | Count |
|----------|-------|
| Universally hard (all 3 models) | 0 |
| Hard in 2+ models | 6 |
| Model-specific targets | 3 |

---

## Universally Hard Questions (All 3 Models)

These questions should be CART's top priority — all models struggle with them.

> No questions are universally hard across all models.

---

## Hard in 2+ Models (Consistent Difficulty)

Questions that multiple models struggle with but not all.

**Q29:** Baraki Barak District is situated in the western part of a province wh

- Ground truth: `Puli Alam`
- **GPT-4o-mini**: gap=100%
- **GPT-5.4-mini**: (not a CART target)
- **Claude Haiku 4.5**: gap=100%

**Q30:** What was the 2010 population of the town where Black Crescent Mountain

- Ground truth: `310`
- **GPT-4o-mini**: gap=100%
- **GPT-5.4-mini**: (not a CART target)
- **Claude Haiku 4.5**: gap=100%

**Q35:** New York State Route 9R rejoins its parent in a hamlet located  in wha

- Ground truth: `Albany`
- **GPT-4o-mini**: gap=67%
- **GPT-5.4-mini**: (not a CART target)
- **Claude Haiku 4.5**: gap=67%

**Q38:** In what city did the "Prince of tenors" star in a film based on an ope

- Ground truth: `Rome`
- **GPT-4o-mini**: gap=100%
- **GPT-5.4-mini**: (not a CART target)
- **Claude Haiku 4.5**: gap=100%

**Q7:** which Mexican and American film actress is Ethel Houbiers  French voic

- Ground truth: `Salma Hayek Pinault`
- **GPT-4o-mini**: gap=80%
- **GPT-5.4-mini**: (not a CART target)
- **Claude Haiku 4.5**: gap=80%

**Q9:** Isabella Kelly was born at a ruined castle characterized as one of the

- Ground truth: `The Changing Scottish Landscape`
- **GPT-4o-mini**: gap=100%
- **GPT-5.4-mini**: (not a CART target)
- **Claude Haiku 4.5**: gap=86%

---

## Model-Specific CART Targets

These are questions where only one model struggles.
They may represent model quirks rather than fundamental difficulty.

### GPT-4o-mini Only (5)

- Q12: `Laurie Metcalf` — Which "Roseanne" star is in Scream 2? (gap=100%)
- Q14: `Cyclic Defrost` — What is the name of the Australian specialist elec (gap=100%)
- Q28: `about 115 miles (185 km)` — How far from Sacramento is the flight school in At (gap=42%)
- Q34: `Jane Mayer` — Who is writing a book about the Koch family who co (gap=100%)
- Q41: `extensive use of segues` — The On Tour Forever album gave Blues Traveler the  (gap=54%)

### GPT-5.4-mini Only (0)

> (None — all targets are shared with other models)

### Claude Haiku 4.5 Only (4)

- Q0: `1755` — In what year was the university where Sergei Aleks (gap=100%)
- Q13: `Nairobi, Kenya` — In what city is the company that Fastjet Tanzania  (gap=67%)
- Q21: `Modern thinkers associated with classical realism are Carl von Clausewitz` — The Prussian General Carl von Clausewitz is associ (gap=41%)
- Q40: `Smoothie King Center` — Jalen Jones plays basketball for an NBA team that  (gap=100%)

---

## Recommendations for CART Implementation

### Priority 1: Universally Hard Questions
- These are your paper's strongest motivation
- Solving even 50% of these shows CART's value
- Use for main paper figures and tables

### Priority 2: Consistent Multi-Model Targets
- Shows CART generalizes across different LLMs
- Good for robustness claims
- Demonstrates the approach isn't model-specific

### Priority 3: Model-Specific Targets (Optional)
- Lower priority for the paper
- Consider skipping if time is limited
- Useful for appendix or future work

### Implementation Strategy
1. **Start with Priority 1** — solve universally hard questions
2. **Validate on Priority 2** — ensure generalization to other models
3. **Extend to Priority 3** — if time permits
