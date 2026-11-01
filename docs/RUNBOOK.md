# Запуск и измерения

Официальный starter зафиксирован на `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`.
`harness/`, `task/` и `tests/` не изменяем.

## Kaggle

1. Импортировать [notebook](../templates/kaggle_notebook.ipynb).
2. GPU T4 x2, Internet on; нужен phone-verified Kaggle account.
3. Для private repo разрешить notebook читать secret `GITHUB_TOKEN` с read-only
   доступом к этому репозиторию. Секрет не писать в git или notebook source.
4. URL команды уже заполнен. В разработке `COMMIT = ""` означает default branch;
   для проверки исследовательской ветки указать её точный commit SHA.
5. Нажать Run All. Шаблон настроен на `--repeats 2`: base, solution, solution, base.
6. Сохранить result/checks и логи до следующего Run All.
7. Для финала указать проверенный SHA, сделать notebook публичным и выполнить
   **Save & Run All**. Сохранить ссылку с `scriptVersionId`.

Ноутбук — шаблон с URL команды, а не опубликованный Kaggle результат.
На starter checks 3 и 4 не проходят: SwiGLU — пример, не собственный kernel.

## Команды в Linux/CUDA окружении

Версии совпадают с официальным notebook:

```bash
python -m pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" \
  "peft==0.20.0" "trl==0.29.1" "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q
python harness/run_task.py --repeats 2
python harness/check_rules.py
python harness/score.py runs/result.json
```

Это команды для подготовки и запуска; пакеты не устанавливаем из `solution.apply()`
или во время замеряемого train. Итоговые результаты принимаются только с Kaggle T4.

Регрессии с патчами отдельно, в том же окружении:

```bash
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
```

Диагностика кандидата:

```bash
python harness/run_task.py --only solution --out runs_try
python harness/run_task.py --profile --out runs_profile
```

Solution-only не имеет базы для ряда checks; не выдавать его за полный допуск.
Profiler — отдельный диагностический запуск, не финальный speedup.

## Что сохранять

Исходные файлы в `runs/` игнорируются git. Выбранные небольшие evidence-файлы
копируем в `evidence/<tested-sha>/<experiment>/`:

- `result.json`, соответствующий `checks.json`;
- train/eval logs, необходимые для проверки выводов;
- `README.md`: commit, notebook/version URL, GPU, команда, версии, вид кандидата,
  failed checks и пределы применимости.

Не коммитим compiler caches, модели, данные, adapters, секреты или гигабайтные traces.
JSON не редактируем; пояснения пишем отдельно. Сохраняем baseline из той же сессии.

## Как читать результат

- Время: wall time всей `soup train`, включая компиляцию; медиана повторов.
- Память: максимальный nvidia-smi peak по повторам, MiB; не PyTorch allocated.
- Качество: официальный eval loss адаптера без наших патчей.
- Собственный вклад: ≥5% wall time шагов 3+; минимум по measured runs.
- Сравнение speedup = base time / ours; memory ratio = base peak / ours.
- Выигрыш <5% по показателю не даёт баллов за этот показатель.
- Все 10 checks обязательны; результаты на других GPU не подтверждают конкурсный выигрыш.

## Финальный проход

1. Свести только проверенные изменения `solution/`, прогнать unit/gradient tests.
2. Запустить официальный ABBA, check_rules без skip-tests и score.
3. Проверить 10/10, собственный вклад, метрики, rewind, форму адаптера и eval.
4. Закрепить tested SHA и сохранить evidence.
5. Заполнить README и SUBMISSION реальными ссылками, авторами и числами.
6. Выполнить публичный Save & Run All с тем же SHA; проверить доступ жюри.
7. Отправить repo, notebook, видео ≤3 минут и презентацию на платформу.

Дедлайн: **31 октября 2026, 23:59 Астана (UTC+5)**.
