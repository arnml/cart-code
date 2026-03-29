# CART Component Ablation — Results (Table 5)

Model         Method            F1  Tokens     Eff
====================================================
gpt4o_mini    cart_base      0.669     449  0.1159
gpt4o_mini    cart_full      0.512     346  0.1001
gpt4o_mini    cart_noise     0.647     619  0.1023
gpt54_mini    cart_base      0.690     451  0.1203
gpt54_mini    cart_full      0.605     257  0.1301
gpt54_mini    cart_noise     0.711     620  0.1131

## CART-Full Routing (key proof of concept)

- gpt4o_mini: think=27 (54%)  retrieve=23 (46%)
- gpt54_mini: think=32 (64%)  retrieve=18 (36%)