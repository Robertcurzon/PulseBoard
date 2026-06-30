# PulseBoard CSV Upload Format

PulseBoard can run on its built-in mock SaaS dataset or on an uploaded CSV. The CSV path is designed to be forgiving so you can show the app with real, anonymized, or spreadsheet-exported data.

## Minimum Required Format

Required:

- `date`: daily or weekly date, parseable by pandas, such as `2026-01-31`
- At least one useful KPI column: `dau`, `mrr`, `new_signups`, `churn_rate`, or `nps`

Recommended:

- `segment`: customer segment, such as `Enterprise`, `Mid-Market`, `Startup`
- `region`: region, such as `North America`, `EMEA`, `APAC`
- `acquisition_channel`: source/channel, such as `Product-Led`, `Paid Search`, `Partner`
- `dau`
- `mau`
- `new_signups`
- `activated_users`
- `paid_conversions`
- `paid_accounts`
- `churned_accounts`
- `mrr`
- `pipeline_created`
- `pipeline_won`
- `expansion_mrr`
- `contraction_mrr`
- `churn_rate`
- `activation_rate`
- `trial_to_paid_rate`
- `net_revenue_retention`
- `feature_a_adoption`
- `feature_b_adoption`
- `nps`
- `retention_d7`
- `retention_d30`

Optional event annotation columns:

- `event`
- `event_category`
- `event_description`

## Notes

- Rates can be entered as decimals, for example `0.041` for 4.1%.
- Missing optional metrics are derived where possible.
- If `segment`, `region`, or `acquisition_channel` are missing, PulseBoard assigns default values so the dashboard still works.
- The app is safest to share with anonymized or mock data. Do not upload customer-identifiable data to a public deployment.

## Tiny Example

```csv
date,segment,region,acquisition_channel,dau,new_signups,mrr,churn_rate,nps,pipeline_created,event
2026-01-01,Enterprise,North America,Partner,4200,180,710000,0.018,58,220000,
2026-01-02,Enterprise,North America,Partner,4350,191,719500,0.017,59,230000,
2026-01-03,Mid-Market,EMEA,Product-Led,3100,240,285000,0.031,48,97000,Product launch
```
