# CART Component Ablation for claude_haiku_4_5 — Results (Global Metrics)

Model         Method            F1  Tokens   Reduct%   Eff_Glb
==================================================================
claude_haiku_4_5cart_base      0.533     456      0.0%    0.0870
claude_haiku_4_5cart_full      0.426     484     -6.1%    0.0688
claude_haiku_4_5cart_noise     0.597     649    -42.3%    0.0922

## CART-Full Routing (key proof of concept)

- claude_haiku_4_5: think=8 (40%)  retrieve=12 (60%)