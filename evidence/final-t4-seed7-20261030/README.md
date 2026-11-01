# Дополнительная проверка seed 7

Тот же source `bef6d7a99779593bcd80d43df013ce34561d55b9`, что у `reserve-v2`.
Официальный harness, seed 7, **одна пара base/solution**, без profiler.
Это проверка устойчивости; основное ABBA-доказательство хранится
[отдельно](../final-t4-20261030/README.md).

| Метрика | Base | Solution |
|---|---:|---:|
| Полная команда, с | 74.67 | 65.10 |
| nvidia-smi peak, MiB | 14351 | 3433 |
| Eval loss | 2.05118 | 2.05361 |
| Own share, шаги 3+ | — | 18.1% |
| Official checks | — | **10/10 PASS** |

Eval delta +0.118%; first loss gap 0.02%; mean curve gap 0.12%.

- [Result](result.json), [checks](checks.json), [verified summary](verified-summary.json).
- [Raw ZIP](raw-evidence.zip): 23 файла, все manifest hashes проверены.
- Archive SHA256: `15fd407f84d68c2480b514a12e49368b2e69928187f5b31337ce0a6c8f23768e`.
- Result SHA256: `5e25d3173d9d80895248e82287937700e32cf60bb443f55e64acf415971ffbb7`.
- Solution SHA256: `4f1789966593fd5c1e37f36e6fbd0ea7735e34d0a20b8ba811a556699f41f193`.

Все 34 solution-файла побайтово сверены с git. Данные задачи, веса, adapters,
compiler caches и credentials в archive не входят. Скрытые строки жюри не проверены.
