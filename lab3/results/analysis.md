Хеш split.csv однаковий у всіх 7 журналах і збігається з поточним: d3a611aec41f6e335ab58287f10a443f3e0e35aa9adfefae662d48b15af567f9
Коміти в журналах: [('07c2cfef6aa3129b64a0e32d7437dd4fd0027a46', True)]; різних версій коду (SHA-256 файлів): 1

### Сім запусків

| # | Режим | Зерно | Втрата train, епоха 8 | macro-F1 validation, епоха 8 | Час навчання, с |
|---|---|---|---|---|---|
| 1 | scratch | 1 | 0,9869 | 0,5955 | 530 |
| 2 | frozen | 1 | 0,1759 | 0,8829 | 297 |
| 3 | frozen | 2 | 0,1848 | 0,8814 | 268 |
| 4 | frozen | 3 | 0,1788 | 0,8864 | 296 |
| 5 | unfreeze | 1 | 0,0908 | 0,8735 | 327 |
| 6 | unfreeze | 2 | 0,0551 | 0,8880 | 364 |
| 7 | unfreeze | 3 | 0,1083 | 0,8469 | 330 |

Пристрій навчання: cpu (Intel64 Family 6 Model 186 Stepping 2, GenuineIntel, 8 threads)

Криві: figures/curves.png

### Розклад ефектів

| Величина | Значення |
|---|---|
| F̄_frozen, R_frozen | 0,8836, 0,0051 |
| F̄_unfreeze, R_unfreeze | 0,8694, 0,0411 |
| F_scratch,1 | 0,5955 |
| R | 0,0411 |
| Δpre | +0,2880 — вартий перевірки (|Δ| > R) |
| Δunf | −0,0141 — не відрізняється від розкиду (|Δ| ≤ R) |
| Δpre + Δunf | +0,2739 |

|Δpre|/R = 7.0, |Δunf|/R = 0.34

### Вибір конфігурації (лише за validation)

Середні macro-F1 validation: scratch 0,5955, frozen 0,8836, unfreeze 0,8694
Правило вибору за журналами → frozen_seed3 (F = 0,8864); lab3_model.pt → frozen_seed3
Перевірка ваг: max |логіти val (CPU) − збережені| = 0.00e+00, збіг argmax 1.0000, macro-F1 val на CPU 0,8864
|Δunf| ≤ R: вибір між frozen і unfreeze визначив шум.

### Тестова оцінка (CPU, один прохід)

macro-F1 test = 0,8892; accuracy test p = 0,8862 (n = 1502); SE = sqrt(p(1−p)/n) = 0,0082
macro-F1 validation вибраного запуску = 0,8864; test − validation = +0,0028

Матриця плутанини (рядки — справжній клас, стовпці — передбачений):

| справжній \ передбачений | buildings | forest | glacier | mountain | sea | street | усього |
|---|---|---|---|---|---|---|---|
| **buildings** | 204 | 1 | 1 | 1 | 1 | 11 | 219 |
| **forest** | 0 | 236 | 0 | 0 | 0 | 1 | 237 |
| **glacier** | 1 | 1 | 207 | 63 | 3 | 2 | 277 |
| **mountain** | 1 | 0 | 33 | 224 | 5 | 0 | 263 |
| **sea** | 3 | 1 | 4 | 5 | 242 | 0 | 255 |
| **street** | 31 | 0 | 0 | 1 | 1 | 218 | 251 |

### П'ять характерних помилок

| Файл | Справжній клас | Передбачений клас | Імовірність передбаченого класу | Кількість таких помилок у test |
|---|---|---|---|---|
| 20179.jpg | glacier | mountain | 0,9199 | 63 |
| 20116.jpg | mountain | glacier | 0,5996 | 33 |
| 20353.jpg | street | buildings | 0,8640 | 31 |
| 20752.jpg | buildings | street | 0,7497 | 11 |
| 20304.jpg | mountain | sea | 0,7211 | 5 |

Рисунки: figures/confusion_matrix.png, figures/test_errors.png
