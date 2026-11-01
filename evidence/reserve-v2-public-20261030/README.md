# Public Save & Run All — reserve-v2

[Kaggle version 363913983](https://www.kaggle.com/code/koko667/setww-reserve-v2-t4?scriptVersionId=363913983)
завершилась успешно за 599 с, включая установку, подготовку, tests и полный ABBA.
Для scoring используется только полная команда обучения, измеренная official harness.
Notebook публичный; source ровно `reserve-v2` / `bef6d7a99779593bcd80d43df013ce34561d55b9`.

| Метрика official ABBA | Base | Solution |
|---|---:|---:|
| Median full command, с | 80.06 | 69.95 |
| Max nvidia-smi peak, MiB | 14345 | 3469 |
| Public eval loss | 2.04965 | 2.05207 |
| Min own share, шаги 3+ | — | 17.7% |
| Checks | — | **10/10 PASS** |

Base wall: 79.00 / 81.12 с; solution: 69.58 / 70.32 с. Пустые compiler caches.
Eval delta +0.118%. 211 tests прошли на T4, включая 79 Soup regressions.
Скрытые данные жюри не проверены. Это новая сессия того же source;
[предыдущий research ABBA](../final-t4-20261030/README.md) сохранён отдельно.

- [Result](result.json), [checks](checks.json), [summary](verified-summary.json), [test log](integration_tests.log).
- [Raw ZIP](raw-evidence.zip) SHA256: `280917afebcd5e50a462e84b6e5162a5836248f15dd1ca87af11102274d3fac8`.
- Result SHA256: `35e728ac41488237861ac98045af75232221aa3d775c6f92d5f42fce6192837d`.
- Solution SHA256: `4f1789966593fd5c1e37f36e6fbd0ea7735e34d0a20b8ba811a556699f41f193`.
- Все 52 записи SOURCE_MANIFEST сверены с git, включая 34 solution-файла.
- Result/checks/provenance связаны одним result hash.

ZIP содержит JSON, config, logs и rewind metadata. Веса, adapters, task rows,
compiler caches и credentials в ZIP не входят.
