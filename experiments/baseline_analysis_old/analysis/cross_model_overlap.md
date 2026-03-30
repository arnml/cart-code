# Cross-Model Analysis: CART Targets Overlap

**Generated:** 2026-03-29 09:26:47

This analysis identifies which CART targets are consistent across models,
helping prioritize the most important questions for CART to solve.

---

## Summary

| Category | Count |
|----------|-------|
| Universally hard (all 3 models) | 2 |
| Hard in 2+ models | 8 |
| Model-specific targets | 6 |

---

## Universally Hard Questions (All 3 Models)

These questions should be CART's top priority — all models struggle with them.

**Q30:** What was the 2010 population of the town where Black Crescent Mountain

- Ground truth: `310`
- **GPT-4o-mini**: think F1=0.000, k5 F1=1.000 (gap=100%)
- **GPT-5.4-mini**: think F1=0.000, k5 F1=1.000 (gap=100%)
- **Claude Haiku 4.5**: think F1=0.000, k5 F1=1.000 (gap=100%)

**Q9:** Isabella Kelly was born at a ruined castle characterized as one of the

- Ground truth: `The Changing Scottish Landscape`
- **GPT-4o-mini**: think F1=0.000, k5 F1=1.000 (gap=100%)
- **GPT-5.4-mini**: think F1=0.000, k5 F1=0.857 (gap=86%)
- **Claude Haiku 4.5**: think F1=0.000, k5 F1=0.857 (gap=86%)

---

## Hard in 2+ Models (Consistent Difficulty)

Questions that multiple models struggle with but not all.

**Q13:** In what city is the company that Fastjet Tanzania was originally found

- Ground truth: `Nairobi, Kenya`
- **GPT-4o-mini**: (not a CART target)
- **GPT-5.4-mini**: gap=100%
- **Claude Haiku 4.5**: gap=100%

**Q14:** What is the name of the Australian specialist electronic music magazin

- Ground truth: `Cyclic Defrost`
- **GPT-4o-mini**: gap=100%
- **GPT-5.4-mini**: (not a CART target)
- **Claude Haiku 4.5**: gap=84%

**Q29:** Baraki Barak District is situated in the western part of a province wh

- Ground truth: `Puli Alam`
- **GPT-4o-mini**: gap=100%
- **GPT-5.4-mini**: (not a CART target)
- **Claude Haiku 4.5**: gap=100%

**Q34:** Who is writing a book about the Koch family who control the second-lar

- Ground truth: `Jane Mayer`
- **GPT-4o-mini**: gap=100%
- **GPT-5.4-mini**: gap=100%
- **Claude Haiku 4.5**: (not a CART target)

**Q35:** New York State Route 9R rejoins its parent in a hamlet located  in wha

- Ground truth: `Albany`
- **GPT-4o-mini**: gap=67%
- **GPT-5.4-mini**: (not a CART target)
- **Claude Haiku 4.5**: gap=67%

**Q47:** What university did the last Detroit Pistons player to wear the number

- Ground truth: `Georgetown University`
- **GPT-4o-mini**: (not a CART target)
- **GPT-5.4-mini**: gap=71%
- **Claude Haiku 4.5**: gap=100%

---

## Model-Specific CART Targets

These are questions where only one model struggles.
They may represent model quirks rather than fundamental difficulty.

### GPT-4o-mini Only (6)

- Q12: `Laurie Metcalf` — Which "Roseanne" star is in Scream 2? (gap=100%)
- Q28: `about 115 miles (185 km)` — How far from Sacramento is the flight school in At (gap=42%)
- Q38: `Rome` — In what city did the "Prince of tenors" star in a  (gap=100%)
- Q41: `extensive use of segues` — The On Tour Forever album gave Blues Traveler the  (gap=54%)
- Q6: `John André` — Who was hung for assisting the attempted surrender (gap=100%)
- ... and 1 more

### GPT-5.4-mini Only (1)

- Q23: `Norwegian language` — what language did the ethnic group which Torstein  (gap=100%)

### Claude Haiku 4.5 Only (5)

- Q0: `1755` — In what year was the university where Sergei Aleks (gap=100%)
- Q21: `Modern thinkers associated with classical realism are Carl von Clausewitz` — The Prussian General Carl von Clausewitz is associ (gap=58%)
- Q27: `the Cold War (194791)` — During what war were the Russia-United Kingdom rel (gap=61%)
- Q36: `Chiwetel Ejiofor` — 12 Years a Slave starred what British actor born 1 (gap=97%)
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
