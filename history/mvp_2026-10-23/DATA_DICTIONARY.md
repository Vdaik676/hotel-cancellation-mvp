# Словарь данных и правила входа

Момент применения: регистрация брони по доступным параметрам. Источник — итоговый исторический CSV; временная доступность значений не подтверждена полностью.

| Поле | Смысл | Проверка |
|---|---|---|
| lead_time | Дней от регистрации до заезда | Целое [0, 1000] |
| stays_in_weekend_nights | Ночей в выходные | Целое [0, 365] |
| stays_in_week_nights | Ночей в будни | Целое [0, 365] |
| adults | Взрослых | Целое [0, 20] |
| children | Детей | Целое [0, 20] |
| babies | Младенцев | Целое [0, 20] |
| hotel | Тип отеля | City Hotel, Resort Hotel |
| arrival_date_month | Месяц заезда | January, February, March, April, May, June, July, August, September, October, November, December |
| meal | Тип питания | BB, FB, HB, SC, Undefined |
| market_segment | Сегмент рынка | Aviation, Complementary, Corporate, Direct, Groups, Offline TA/TO, Online TA |
| reserved_room_type | Забронированный тип номера | A, B, C, D, E, F, G, H, L |

Всего ночей 1–365; гостей 1–20. lead_time ограничен 0–1000 как учебный предел. Пропущенные и дробные числовые значения не принимаются формой. Четыре исходных пропуска children исключены на этапе качества.

Цель is_canceled (0/1) используется только в обучении/оценке. Все остальные исходные поля не входят в X. Явно запрещены reservation_status, reservation_status_date и сам is_canceled. ADR, изменения брони, назначенный номер, депозит и спецзапросы не включены до проверки временной доступности.

Препроцессор: медианная импутация и StandardScaler для чисел; OneHotEncoder для категорий. fit только на train. Offline encoder handle_unknown=ignore; ручной UI отклоняет неизвестные категории.

Источник: https://github.com/rfordatascience/tidytuesday/blob/main/data/2020/2020-02-11/readme.md