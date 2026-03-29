# CART Component Ablation — Results (Global Metrics)

Model         Method            F1  Tokens   Reduct%   Eff_Glb
==================================================================
gpt4o_mini    cart_base      0.642     418      0.0%    0.1064
gpt4o_mini    cart_full      0.474     340     18.6%    0.0813
gpt4o_mini    cart_noise     0.649     591    -41.6%    0.1017
gpt54_mini    cart_base      0.657     420      0.0%    0.1088
gpt54_mini    cart_full      0.585     277     34.0%    0.1040
gpt54_mini    cart_noise     0.704     593    -41.4%    0.1103

## CART-Full Routing (key proof of concept)

- gpt4o_mini: think=21 (52%)  retrieve=19 (48%)
- gpt54_mini: think=24 (60%)  retrieve=16 (40%)