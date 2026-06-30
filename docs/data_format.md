# PulseBoard CSV Upload Format

PulseBoard can run on its built-in mock SaaS dataset or on an uploaded CSV. The CSV path is designed to be forgiving enough for spreadsheet exports while still producing useful filters, forecasts, anomaly detection, and AI analyst output.

## Minimum Required Format

Required:

- `date`: daily or weekly date, parseable by pandas, such as `2026-01-31`
- At least one useful KPI column: `dau`, `mrr`, `new_signups`, `churn_rate`, or `nps`

Recommended dimensions:

- `segment`: customer segment, such as `Enterprise`, `Mid-Market`, `Startup`
- `region`: region, such as `North America`, `EMEA`, `APAC`
- `acquisition_channel`: source/channel, such as `Product-Led`, `Paid Search`, `Partner`
Recommended KPI and operating metrics:

- `dau`: daily active users
- `mau`: monthly active users
- `new_signups`: new accounts/users entering the funnel
- `activated_users`: users reaching an activation milestone
- `paid_conversions`: users/accounts converting to paid
- `paid_accounts`: current paid account count
- `churned_accounts`: accounts that churned in the period
- `mrr`: monthly recurring revenue
- `arpu`: average revenue per user/account
- `pipeline_created`: pipeline value created in the period
- `pipeline_won`: pipeline value won in the period
- `expansion_mrr`: expansion revenue
- `contraction_mrr`: contraction or downgrade revenue
- `churn_rate`: decimal churn rate
- `activation_rate`: decimal activation rate
- `trial_to_paid_rate`: decimal trial-to-paid conversion rate
- `net_revenue_retention`: decimal NRR, such as `1.04` for 104%
- `feature_a_adoption`: decimal feature adoption rate
- `feature_b_adoption`: decimal feature adoption rate
- `nps`: Net Promoter Score
- `retention_d7`: decimal day-7 retention
- `retention_d30`: decimal day-30 retention

Optional event annotation columns:

- `event`
- `event_category`
- `event_description`

## Notes

- Rates can be entered as decimals, for example `0.041` for 4.1%.
- Missing optional metrics are derived where possible.
- If `segment`, `region`, or `acquisition_channel` are missing, PulseBoard assigns default values so the dashboard still works.
- For forecasting, upload at least 45 dated rows when possible. Shorter files still work, but the app uses a simpler baseline forecast.
- The app is safest to share with anonymized or mock data. Do not upload customer-identifiable data to a public deployment.

## Accepted Header Aliases

PulseBoard normalizes common spreadsheet-friendly names:

| CSV header | Normalized field |
| --- | --- |
| `day`, `period` | `date` |
| `customer_segment`, `market_segment` | `segment` |
| `geo`, `country_region` | `region` |
| `channel`, `source` | `acquisition_channel` |
| `revenue`, `monthly_recurring_revenue` | `mrr` |
| `active_users`, `daily_active_users` | `dau` |
| `signup`, `signups` | `new_signups` |
| `trial_to_paid`, `conversion_rate` | `trial_to_paid_rate` |
| `nrr` | `net_revenue_retention` |
| `feature_a`, `feature_b` | `feature_a_adoption`, `feature_b_adoption` |

## Compatibility Checklist

- Use one row per date, segment, region, and acquisition channel combination when possible.
- Keep dates parseable, for example `2026-01-31`.
- Use numbers without currency symbols or percent signs.
- Use decimals for rates.
- Include `event`, `event_category`, and `event_description` when you want anomaly narratives to reference launches, incidents, campaigns, or operational changes.

## Tiny Example

```csv
date,segment,region,acquisition_channel,dau,new_signups,mrr,churn_rate,nps,pipeline_created,event
2026-01-01,Enterprise,North America,Partner,4200,180,710000,0.018,58,220000,
2026-01-02,Enterprise,North America,Partner,4350,191,719500,0.017,59,230000,
2026-01-03,Mid-Market,EMEA,Product-Led,3100,240,285000,0.031,48,97000,Product launch
```
