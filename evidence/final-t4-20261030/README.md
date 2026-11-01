# Verified T4 ABBA — bef6d7a9

| Официальный ABBA | Base | Solution |
|---|---:|---:|
| Медиана полной команды, с | 79.94 | 68.09 |
| Максимальный nvidia-smi peak, MiB | 14345 | 3469 |
| Public eval loss | 2.04966 | 2.05219 |
| Минимальная авторская доля, шаги 3+ | — | 17.8593% |
| Official checks | — | **10/10 PASS** |

ABBA wall times: 77.38 / 67.41 / 68.76 / 82.49 s; fresh compiler caches for each command.
211 T4 tests and all ten official checks, including 79 Soup regressions.

- Source `bef6d7a99779593bcd80d43df013ce34561d55b9`: 34 solution files verified byte-for-byte against git.
- Solution SHA256: `4f1789966593fd5c1e37f36e6fbd0ea7735e34d0a20b8ba811a556699f41f193`.
- Result SHA256: `b1ab8698a3149919276153404e37a0c9a6707ed601c41e2649438448141ea447`.
- [Raw ZIP](raw-evidence.zip) SHA256: `d65a8681ad87493bd7d1b5edabe8c2f7c005fc5aa769d0360b2f5dc1db496f9a`.
- All 297 ZIP files verified against EVIDENCE_MANIFEST.json.
- [Result](result.json), [checks](checks.json), [verified summary](verified-summary.json),
  [measured source](solution-source.zip), [test log](integration_tests.log).

The ZIP contains all 15 research cases plus prior reserve reproduction, with JSON,
logs, profiles and exact source snapshots. Weights, task rows, compiler caches,
databases and credentials are excluded. Screening cases are not ABBA claims.
Offload, attention recompute and runtime compile overlap are inactive in this source.
Original starter SwiGLU is excluded from own share. Actual versions are in result.json.
