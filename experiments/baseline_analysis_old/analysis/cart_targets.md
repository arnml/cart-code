# CART Targets: Baseline Diagnostics

**Generated:** 2026-03-29 09:26:46

This analysis identifies questions where CART needs to excel:
- **CART Targets**: Questions where think fails but retrieval succeeds
- **Token Usage**: How different models use tokens on think-only tasks
- **Summary Stats**: Baseline performance across methods

---


## GPT-4o-mini

### CART Targets: Questions Retrieval Solves (12)

Questions where think F1 < 0.3 but k5 F1 > 0.6 (retrieval is crucial)

**Q6:** Who was hung for assisting the attempted surrender of a defector from 

- Ground truth: `John André`
- Think F1: **0.000** (213 tokens)
- K5 F1: **1.000** (880 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q9:** Isabella Kelly was born at a ruined castle characterized as one of the

- Ground truth: `The Changing Scottish Landscape`
- Think F1: **0.000** (187 tokens)
- K5 F1: **1.000** (681 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q12:** Which "Roseanne" star is in Scream 2?

- Ground truth: `Laurie Metcalf`
- Think F1: **0.000** (270 tokens)
- K5 F1: **1.000** (919 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q14:** What is the name of the Australian specialist electronic music magazin

- Ground truth: `Cyclic Defrost`
- Think F1: **0.000** (166 tokens)
- K5 F1: **1.000** (593 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q29:** Baraki Barak District is situated in the western part of a province wh

- Ground truth: `Puli Alam`
- Think F1: **0.000** (134 tokens)
- K5 F1: **1.000** (613 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q30:** What was the 2010 population of the town where Black Crescent Mountain

- Ground truth: `310`
- Think F1: **0.000** (193 tokens)
- K5 F1: **1.000** (573 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q34:** Who is writing a book about the Koch family who control the second-lar

- Ground truth: `Jane Mayer`
- Think F1: **0.000** (187 tokens)
- K5 F1: **1.000** (802 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q38:** In what city did the "Prince of tenors" star in a film based on an ope

- Ground truth: `Rome`
- Think F1: **0.000** (161 tokens)
- K5 F1: **1.000** (604 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q7:** which Mexican and American film actress is Ethel Houbiers  French voic

- Ground truth: `Salma Hayek Pinault`
- Think F1: **0.000** (217 tokens)
- K5 F1: **0.800** (519 tokens)
- Retrieval gap: **+80.0%** (CART should close this)

**Q35:** New York State Route 9R rejoins its parent in a hamlet located  in wha

- Ground truth: `Albany`
- Think F1: **0.000** (167 tokens)
- K5 F1: **0.667** (1060 tokens)
- Retrieval gap: **+66.7%** (CART should close this)

### Sample Think-Only Outputs

**Sample 1:**

- **Q:** In what year was the university where Sergei Aleksandrovich 
- **Ground truth:** `1755`
- **Answer:** `1755...`
- **F1:** 1.0 | **Tokens:** 147

**Sample 2:**

- **Q:** Black Book starred the actress and writer of what heritage?
- **Ground truth:** `Dutch`
- **Answer:** `Dutch...`
- **F1:** 1.0 | **Tokens:** 136

**Sample 3:**

- **Q:** Which actor does American Beauty and American Beauty have in
- **Ground truth:** `Kevin Spacey`
- **Answer:** `Kevin Spacey...`
- **F1:** 1.0 | **Tokens:** 136

**Sample 4:**

- **Q:** Ken Pruitt  was a Republican member of an upper house of the
- **Ground truth:** `40 members`
- **Answer:** `40 members...`
- **F1:** 1.0 | **Tokens:** 123

**Sample 5:**

- **Q:** Between Greyia and Calibanus, which genus contains more spec
- **Ground truth:** `Greyia`
- **Answer:** `Greyia contains more species than Calibanus....`
- **F1:** 0.2857 | **Tokens:** 201

### Summary Statistics

| Method | Avg F1 | Avg Tokens | Avg Cost |
|--------|--------|-----------|----------|
| always_retrieve_k3 | 0.6294 | 492 | $0.00011 |
| always_retrieve_k5 | 0.7065 | 752 | $0.00015 |
| always_think | 0.4383 | 176 | $0.00007 |


### GPT-5.4-mini

⚠️ No results found at `C:\code\smtl-code\experiments\baseline_analysis\results\results_gpt_5.4_mini_2026_03_17\results.csv`


## Claude Haiku 4.5

### CART Targets: Questions Retrieval Solves (12)

Questions where think F1 < 0.3 but k5 F1 > 0.6 (retrieval is crucial)

**Q0:** In what year was the university where Sergei Aleksandrovich Tokarev wa

- Ground truth: `1755`
- Think F1: **0.000** (227 tokens)
- K5 F1: **1.000** (723 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q13:** In what city is the company that Fastjet Tanzania was originally found

- Ground truth: `Nairobi, Kenya`
- Think F1: **0.000** (306 tokens)
- K5 F1: **1.000** (1019 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q29:** Baraki Barak District is situated in the western part of a province wh

- Ground truth: `Puli Alam`
- Think F1: **0.000** (225 tokens)
- K5 F1: **1.000** (794 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q30:** What was the 2010 population of the town where Black Crescent Mountain

- Ground truth: `310`
- Think F1: **0.000** (309 tokens)
- K5 F1: **1.000** (767 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q40:** Jalen Jones plays basketball for an NBA team that plays their home gam

- Ground truth: `Smoothie King Center`
- Think F1: **0.000** (282 tokens)
- K5 F1: **1.000** (746 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q47:** What university did the last Detroit Pistons player to wear the number

- Ground truth: `Georgetown University`
- Think F1: **0.000** (390 tokens)
- K5 F1: **1.000** (1107 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q36:** 12 Years a Slave starred what British actor born 10 July 1977)

- Ground truth: `Chiwetel Ejiofor`
- Think F1: **0.025** (381 tokens)
- K5 F1: **1.000** (833 tokens)
- Retrieval gap: **+97.5%** (CART should close this)

**Q9:** Isabella Kelly was born at a ruined castle characterized as one of the

- Ground truth: `The Changing Scottish Landscape`
- Think F1: **0.000** (384 tokens)
- K5 F1: **0.857** (815 tokens)
- Retrieval gap: **+85.7%** (CART should close this)

**Q14:** What is the name of the Australian specialist electronic music magazin

- Ground truth: `Cyclic Defrost`
- Think F1: **0.160** (266 tokens)
- K5 F1: **1.000** (747 tokens)
- Retrieval gap: **+84.0%** (CART should close this)

**Q35:** New York State Route 9R rejoins its parent in a hamlet located  in wha

- Ground truth: `Albany`
- Think F1: **0.000** (236 tokens)
- K5 F1: **0.667** (1180 tokens)
- Retrieval gap: **+66.7%** (CART should close this)

### Sample Think-Only Outputs

**Sample 1:**

- **Q:** In what year was the university where Sergei Aleksandrovich 
- **Ground truth:** `1755`
- **Answer:** `1724...`
- **F1:** 0.0 | **Tokens:** 227

**Sample 2:**

- **Q:** Black Book starred the actress and writer of what heritage?
- **Ground truth:** `Dutch`
- **Answer:** `I cannot provide a definitive answer as the premise appears to be inaccurate - the actress in "Black...`
- **F1:** 0.0 | **Tokens:** 372

**Sample 3:**

- **Q:** Which actor does American Beauty and American Beauty have in
- **Ground truth:** `Kevin Spacey`
- **Answer:** `This question appears to contain a redundancy as it mentions "American Beauty" twice. If asking abou...`
- **F1:** 0.1 | **Tokens:** 349

**Sample 4:**

- **Q:** Ken Pruitt  was a Republican member of an upper house of the
- **Ground truth:** `40 members`
- **Answer:** `40...`
- **F1:** 0.6667 | **Tokens:** 194

**Sample 5:**

- **Q:** Between Greyia and Calibanus, which genus contains more spec
- **Ground truth:** `Greyia`
- **Answer:** `Greyia contains more species than Calibanus....`
- **F1:** 0.2857 | **Tokens:** 269

### Summary Statistics

| Method | Avg F1 | Avg Tokens | Avg Cost |
|--------|--------|-----------|----------|
| always_retrieve_k3 | 0.3624 | 648 | $0.00140 |
| always_retrieve_k5 | 0.4910 | 932 | $0.00165 |
| always_think | 0.2516 | 289 | $0.00111 |

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
