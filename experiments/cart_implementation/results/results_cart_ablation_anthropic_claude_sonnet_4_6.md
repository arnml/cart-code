# CART Component Ablation for claude_sonnet_4_6 — Results (Table 5)

Model         Method            F1  Tokens     Eff
====================================================
claude_sonnet_4_6cart_base      0.721     448  0.1243
claude_sonnet_4_6cart_full      0.623     327  0.1302
claude_sonnet_4_6cart_noise     0.796     642  0.1253

## CART-Full Routing (key proof of concept)

- claude_sonnet_4_6: think=12 (60%)  retrieve=8 (40%)