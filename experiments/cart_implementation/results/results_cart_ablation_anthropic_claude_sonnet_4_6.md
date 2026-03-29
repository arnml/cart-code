# CART Component Ablation for claude_sonnet_4_6 — Results (Global Metrics)

Model         Method            F1  Tokens   Reduct%   Eff_Glb
==================================================================
claude_sonnet_4_6cart_base      0.721     448      0.0%    0.1181
claude_sonnet_4_6cart_full      0.623     327     26.9%    0.1075
claude_sonnet_4_6cart_noise     0.771     642    -43.5%    0.1192

## CART-Full Routing (key proof of concept)

- claude_sonnet_4_6: think=12 (60%)  retrieve=8 (40%)