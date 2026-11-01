## Что меняется и зачем

Связанная задача: #
Направление: R1 / R2 / R3 / интеграция / документация

## Проверка

- Коммит и Kaggle notebook/version:
- Forward/backward и реальные updates:
- Full-command time / nvidia-smi peak / eval delta:
- Own-time share min / failed checks:
- Если T4-замер не выполнялся, явно указать это.

## Перед review

- [ ] `harness/`, `task/`, `tests/` не менялись.
- [ ] Изменения runtime самодостаточны в `solution/`.
- [ ] Метрики и rewind сохраняются; источники идей указаны.
- [ ] Новые результаты имеют evidence, отрицательные результаты не скрыты.
- [ ] Изменения `apply()` согласованы с интегратором.
