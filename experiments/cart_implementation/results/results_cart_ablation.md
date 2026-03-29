# CART Component Ablation — Results (Table 5)

Model         Method            F1  Tokens     Eff
====================================================
gpt4o_mini    cart_base      0.649     449  0.1124
gpt4o_mini    cart_full      0.432      50  0.1096
gpt4o_mini    cart_noise     0.667     619  0.1053
gpt54_mini    cart_base      0.690     451  0.1203
gpt54_mini    cart_full      0.482      52  0.1216
gpt54_mini    cart_noise     0.711     620  0.1131

## CART-Full Routing (key proof of concept)

- gpt4o_mini: think=50 (100%)  retrieve=0 (0%)
- gpt54_mini: think=50 (100%)  retrieve=0 (0%)