# Data-quality report

Source: **synthetic** | calendar: ^GSPC trading days (2015-01-02 to 2026-10-28, 2962 sessions)

| metric | value |
|---|---|
| rows in | 38503 |
| rows out | 38506 |
| duplicates removed | 2 |
| gap sessions filled | 5 |
| gaps longer than 3 flagged | 0 |
| outliers flagged | 69 |
| outliers kept as events | 68 |
| outliers winsorised | 1 |
| ohlc violations | 1 |

**Data contract (PRD 3.4)**

- history ge 8 years: PASS
- universe size 8 to 15: PASS
- zero missing days post clean: PASS
- single trading calendar: PASS

**Per ticker**

| ticker   |   rows_in |   rows_out |   duplicates_removed |   gap_sessions_filled |   outliers_flagged |   outliers_winsorised |   ohlc_violations_quarantined |   missing_days_post_clean |
|:---------|----------:|-----------:|---------------------:|----------------------:|-------------------:|----------------------:|------------------------------:|--------------------------:|
| AAPL     |      2957 |       2962 |                    0 |                     5 |                  6 |                     0 |                             0 |                         0 |
| CAT      |      2962 |       2962 |                    0 |                     0 |                  7 |                     0 |                             0 |                         0 |
| HD       |      2962 |       2962 |                    0 |                     0 |                  3 |                     0 |                             0 |                         0 |
| JNJ      |      2962 |       2962 |                    0 |                     0 |                  8 |                     0 |                             0 |                         0 |
| JPM      |      2962 |       2962 |                    0 |                     0 |                  5 |                     0 |                             1 |                         0 |
| KO       |      2962 |       2962 |                    0 |                     0 |                  6 |                     0 |                             0 |                         0 |
| MSFT     |      2964 |       2962 |                    2 |                     0 |                  2 |                     0 |                             0 |                         0 |
| NVDA     |      2962 |       2962 |                    0 |                     0 |                 11 |                     0 |                             0 |                         0 |
| PG       |      2962 |       2962 |                    0 |                     0 |                  5 |                     0 |                             0 |                         0 |
| XLK      |      2962 |       2962 |                    0 |                     0 |                  2 |                     0 |                             0 |                         0 |
| XOM      |      2962 |       2962 |                    0 |                     0 |                  7 |                     1 |                             0 |                         0 |
| ^GSPC    |      2962 |       2962 |                    0 |                     0 |                  4 |                     0 |                             0 |                         0 |
| ^VIX     |      2962 |       2962 |                    0 |                     0 |                  3 |                     0 |                             0 |                         0 |