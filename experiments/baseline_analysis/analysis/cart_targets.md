# CART Targets: Baseline Diagnostics

**Generated:** 2026-03-28 12:38:04

This analysis identifies questions where CART needs to excel:
- **CART Targets**: Questions where think fails but retrieval succeeds
- **Token Usage**: How different models use tokens on think-only tasks
- **Summary Stats**: Baseline performance across methods

---


## GPT-4o-mini

### CART Targets: Questions Retrieval Solves (11)

Questions where think F1 < 0.3 but k5 F1 > 0.6 (retrieval is crucial)

**Q9:** Isabella Kelly was born at a ruined castle characterized as one of the

- Ground truth: `The Changing Scottish Landscape`
- Think F1: **0.000** (188 tokens)
- K5 F1: **1.000** (683 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q12:** Which "Roseanne" star is in Scream 2?

- Ground truth: `Laurie Metcalf`
- Think F1: **0.000** (149 tokens)
- K5 F1: **1.000** (919 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q14:** What is the name of the Australian specialist electronic music magazin

- Ground truth: `Cyclic Defrost`
- Think F1: **0.000** (167 tokens)
- K5 F1: **1.000** (593 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q29:** Baraki Barak District is situated in the western part of a province wh

- Ground truth: `Puli Alam`
- Think F1: **0.000** (138 tokens)
- K5 F1: **1.000** (613 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q30:** What was the 2010 population of the town where Black Crescent Mountain

- Ground truth: `310`
- Think F1: **0.000** (169 tokens)
- K5 F1: **1.000** (573 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q34:** Who is writing a book about the Koch family who control the second-lar

- Ground truth: `Jane Mayer`
- Think F1: **0.000** (188 tokens)
- K5 F1: **1.000** (805 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q38:** In what city did the "Prince of tenors" star in a film based on an ope

- Ground truth: `Rome`
- Think F1: **0.000** (166 tokens)
- K5 F1: **1.000** (614 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q7:** which Mexican and American film actress is Ethel Houbiers  French voic

- Ground truth: `Salma Hayek Pinault`
- Think F1: **0.000** (188 tokens)
- K5 F1: **0.800** (525 tokens)
- Retrieval gap: **+80.0%** (CART should close this)

**Q35:** New York State Route 9R rejoins its parent in a hamlet located  in wha

- Ground truth: `Albany`
- Think F1: **0.000** (167 tokens)
- K5 F1: **0.667** (1060 tokens)
- Retrieval gap: **+66.7%** (CART should close this)

**Q41:** The On Tour Forever album gave Blues Traveler the opportunity to displ

- Ground truth: `extensive use of segues`
- Think F1: **0.125** (177 tokens)
- K5 F1: **0.667** (510 tokens)
- Retrieval gap: **+54.2%** (CART should close this)

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
- **F1:** 1.0 | **Tokens:** 139

**Sample 4:**

- **Q:** Ken Pruitt  was a Republican member of an upper house of the
- **Ground truth:** `40 members`
- **Answer:** `40 members...`
- **F1:** 1.0 | **Tokens:** 123

**Sample 5:**

- **Q:** Between Greyia and Calibanus, which genus contains more spec
- **Ground truth:** `Greyia`
- **Answer:** `Greyia contains more species than Calibanus....`
- **F1:** 0.2857 | **Tokens:** 194

### Summary Statistics

| Method | Avg F1 | Avg Tokens | Avg Cost |
|--------|--------|-----------|----------|
| always_retrieve_k3 | 0.6247 | 492 | $0.00011 |
| always_retrieve_k5 | 0.7109 | 753 | $0.00015 |
| always_think | 0.4699 | 173 | $0.00007 |


## GPT-5.4-mini

### CART Targets: Questions Retrieval Solves (6)

Questions where think F1 < 0.3 but k5 F1 > 0.6 (retrieval is crucial)

**Q13:** In what city is the company that Fastjet Tanzania was originally found

- Ground truth: `Nairobi, Kenya`
- Think F1: **0.000** (131 tokens)
- K5 F1: **1.000** (814 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q23:** what language did the ethnic group which Torstein Ellingsen was its dr

- Ground truth: `Norwegian language`
- Think F1: **0.000** (138 tokens)
- K5 F1: **1.000** (717 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q30:** What was the 2010 population of the town where Black Crescent Mountain

- Ground truth: `310`
- Think F1: **0.000** (137 tokens)
- K5 F1: **1.000** (561 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q34:** Who is writing a book about the Koch family who control the second-lar

- Ground truth: `Jane Mayer`
- Think F1: **0.000** (142 tokens)
- K5 F1: **1.000** (802 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q38:** In what city did the "Prince of tenors" star in a film based on an ope

- Ground truth: `Rome`
- Think F1: **0.000** (177 tokens)
- K5 F1: **1.000** (607 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q9:** Isabella Kelly was born at a ruined castle characterized as one of the

- Ground truth: `The Changing Scottish Landscape`
- Think F1: **0.000** (141 tokens)
- K5 F1: **0.857** (702 tokens)
- Retrieval gap: **+85.7%** (CART should close this)

### Sample Think-Only Outputs

**Sample 1:**

- **Q:** In what year was the university where Sergei Aleksandrovich 
- **Ground truth:** `1755`
- **Answer:** `1755...`
- **F1:** 1.0 | **Tokens:** 138

**Sample 2:**

- **Q:** Black Book starred the actress and writer of what heritage?
- **Ground truth:** `Dutch`
- **Answer:** `Dutch heritage...`
- **F1:** 0.6667 | **Tokens:** 110

**Sample 3:**

- **Q:** Which actor does American Beauty and American Beauty have in
- **Ground truth:** `Kevin Spacey`
- **Answer:** `Kevin Spacey...`
- **F1:** 1.0 | **Tokens:** 129

**Sample 4:**

- **Q:** Ken Pruitt  was a Republican member of an upper house of the
- **Ground truth:** `40 members`
- **Answer:** `40 members...`
- **F1:** 1.0 | **Tokens:** 127

**Sample 5:**

- **Q:** Between Greyia and Calibanus, which genus contains more spec
- **Ground truth:** `Greyia`
- **Answer:** `Greyia...`
- **F1:** 1.0 | **Tokens:** 127

### Summary Statistics

| Method | Avg F1 | Avg Tokens | Avg Cost |
|--------|--------|-----------|----------|
| always_retrieve_k3 | 0.6755 | 483 | $0.00063 |
| always_retrieve_k5 | 0.7526 | 741 | $0.00081 |
| always_think | 0.5798 | 142 | $0.00035 |


## Claude Haiku 4.5

### CART Targets: Questions Retrieval Solves (10)

Questions where think F1 < 0.3 but k5 F1 > 0.6 (retrieval is crucial)

**Q0:** In what year was the university where Sergei Aleksandrovich Tokarev wa

- Ground truth: `1755`
- Think F1: **0.000** (207 tokens)
- K5 F1: **1.000** (724 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q29:** Baraki Barak District is situated in the western part of a province wh

- Ground truth: `Puli Alam`
- Think F1: **0.000** (175 tokens)
- K5 F1: **1.000** (833 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q30:** What was the 2010 population of the town where Black Crescent Mountain

- Ground truth: `310`
- Think F1: **0.000** (310 tokens)
- K5 F1: **1.000** (823 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q38:** In what city did the "Prince of tenors" star in a film based on an ope

- Ground truth: `Rome`
- Think F1: **0.000** (389 tokens)
- K5 F1: **1.000** (765 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q40:** Jalen Jones plays basketball for an NBA team that plays their home gam

- Ground truth: `Smoothie King Center`
- Think F1: **0.000** (306 tokens)
- K5 F1: **1.000** (752 tokens)
- Retrieval gap: **+100.0%** (CART should close this)

**Q9:** Isabella Kelly was born at a ruined castle characterized as one of the

- Ground truth: `The Changing Scottish Landscape`
- Think F1: **0.000** (353 tokens)
- K5 F1: **0.857** (821 tokens)
- Retrieval gap: **+85.7%** (CART should close this)

**Q7:** which Mexican and American film actress is Ethel Houbiers  French voic

- Ground truth: `Salma Hayek Pinault`
- Think F1: **0.000** (242 tokens)
- K5 F1: **0.800** (630 tokens)
- Retrieval gap: **+80.0%** (CART should close this)

**Q13:** In what city is the company that Fastjet Tanzania was originally found

- Ground truth: `Nairobi, Kenya`
- Think F1: **0.000** (235 tokens)
- K5 F1: **0.667** (996 tokens)
- Retrieval gap: **+66.7%** (CART should close this)

**Q35:** New York State Route 9R rejoins its parent in a hamlet located  in wha

- Ground truth: `Albany`
- Think F1: **0.000** (343 tokens)
- K5 F1: **0.667** (1213 tokens)
- Retrieval gap: **+66.7%** (CART should close this)

**Q21:** The Prussian General Carl von Clausewitz is associated with what type 

- Ground truth: `Modern thinkers associated with classical realism are Carl von Clausewitz`
- Think F1: **0.256** (326 tokens)
- K5 F1: **0.667** (1181 tokens)
- Retrieval gap: **+41.0%** (CART should close this)

### Sample Think-Only Outputs

**Sample 1:**

- **Q:** In what year was the university where Sergei Aleksandrovich 
- **Ground truth:** `1755`
- **Answer:** `1724...`
- **F1:** 0.0 | **Tokens:** 207

**Sample 2:**

- **Q:** Black Book starred the actress and writer of what heritage?
- **Ground truth:** `Dutch`
- **Answer:** `Dutch heritage...`
- **F1:** 0.6667 | **Tokens:** 164

**Sample 3:**

- **Q:** Which actor does American Beauty and American Beauty have in
- **Ground truth:** `Kevin Spacey`
- **Answer:** `If the question is asking about a common actor between two different films and "American Beauty" was...`
- **F1:** 0.0784 | **Tokens:** 298

**Sample 4:**

- **Q:** Ken Pruitt  was a Republican member of an upper house of the
- **Ground truth:** `40 members`
- **Answer:** `40...`
- **F1:** 0.6667 | **Tokens:** 153

**Sample 5:**

- **Q:** Between Greyia and Calibanus, which genus contains more spec
- **Ground truth:** `Greyia`
- **Answer:** `Greyia contains more species than Calibanus (approximately 3 species vs. 2 species)....`
- **F1:** 0.1538 | **Tokens:** 305

### Summary Statistics

| Method | Avg F1 | Avg Tokens | Avg Cost |
|--------|--------|-----------|----------|
| always_retrieve_k3 | 0.3832 | 649 | $0.00141 |
| always_retrieve_k5 | 0.4764 | 935 | $0.00166 |
| always_think | 0.2825 | 279 | $0.00106 |

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
