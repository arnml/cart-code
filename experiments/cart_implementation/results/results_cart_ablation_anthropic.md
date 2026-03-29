# CART Component Ablation for Anthropic — Results (Table 5)

Model         Method            F1  Tokens     Eff
====================================================
claude_haiku  cart_base      0.532     456  0.0930
claude_haiku  cart_full      0.426     484  0.0717
claude_haiku  cart_noise     0.597     650  0.0940

## CART-Full Routing (key proof of concept)

- claude_haiku: think=8 (40%)  retrieve=12 (60%)