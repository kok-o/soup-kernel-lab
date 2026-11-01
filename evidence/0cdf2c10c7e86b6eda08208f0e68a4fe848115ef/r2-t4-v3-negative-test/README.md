# R2 additional cast coverage — failed fixture

Source `0cdf2c10c7e86b6eda08208f0e68a4fe848115ef`, Kaggle saved version 3:
https://www.kaggle.com/code/narimanabdikarim/setww-r2-abba-3527e32/versions/3

CUDA tests stopped before training: 88 passed, 5 failed. First error was
`misaligned address` during NF4 test of a synthetic 128→257 biased layer.
Later SwiGLU failures follow a poisoned CUDA context, not independent findings.
No official training or performance result was produced.

The next fixture uses aligned 128→256 dimensions and synchronizes reference
and candidate separately. It passed with the full 93-test CUDA suite; its
performance is assessed independently. Raw failed tests are preserved here.
