# CART Targets: Baseline Diagnostics

**Generated:** 2026-03-29 09:22:37

This analysis identifies questions where CART needs to excel:
- **CART Targets**: Questions where think fails but retrieval succeeds
- **Token Usage**: How different models use tokens on think-only tasks
- **Summary Stats**: Baseline performance across methods

---


### GPT-4o-mini

⚠️ No results found at `results\results_gpt_4o_mini\results.csv`


### GPT-5.4-mini

⚠️ No results found at `results\results_gpt_5.4_mini_2026_03_17\results.csv`


### Claude Haiku 4.5

⚠️ No results found at `results\results_claude_haiku_4_5\results.csv`

---

## CART Design Implications

### Questions Where Retrieval Genuinely Helps
These are your CART targets. CART should:
- Retrieve documents for these questions (like k5 does)
- Close the F1 gap between think and k5
- Use fewer tokens than k5 does it

### Token Usage Analysis
Compare think tokens across models:
- **GPT-4o-mini:** Baseline for think-only reasoning
- **GPT-5.4-mini:** Higher tokens might indicate longer reasoning
- **Haiku:** If tokens are higher than GPT-4o-mini, it may be:
  - More verbose reasoning
  - Different prompt interpretation
  - Padding or system message differences

### Performance Patterns
- If k5 doesn't help much: embedding retrieval is noisy
  - CART might need better re-ranking or filtering
- If k5 helps significantly: retrieval is valuable
  - CART should adaptively retrieve only when needed

### CART Success Criteria

For questions in CART Targets:
```
CART F1 >= always_retrieve_k5 F1
CART tokens < always_retrieve_k5 tokens (50%+ improvement?)
CART efficiency > always_retrieve_k5 efficiency
```

For other questions:
```
CART should be competitive with always_think
(low-cost fallback when retrieval not needed)
```
