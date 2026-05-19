# Risk Analytics Case Study — Data Dictionary
This synthetic dataset is de-identified and created for interview case study purposes. It is designed to reflect real-world data imperfections and cross-table inconsistencies.
## File: `interactions.csv`
| Column | Type | Description | Example | Notes |
|---|---|---|---|---|
| interaction_id | string | Interaction record identifier | I000125 | |
| interaction_date | string (date-like) | Date of the interaction | 2024-09-14 | |
| country | string | Country where interaction occurred | US / United States / DE | |
| region | string | Region grouping | EMEA | |
| business_unit | string | Business area | Vaccines | Categorical |
| interaction_type | string | Type of interaction | 1:1 Meeting | Categorical |
| stakeholder_id | string | External stakeholder identifier | S00421 | Foreign key to `stakeholders.csv` |
| employee_id | string | Internal employee identifier | E00110 | Foreign key to `employees.csv` |
| declared_purpose | string | Short text purpose description | Scientific exchange on clinical data | |
| duration_minutes | integer | Duration in minutes | 60 | |

## File: `spend.csv`
| Column | Type | Description | Example | Notes |
|---|---|---|---|---|
| spend_id | string | Spend record identifier | P000042 | Unique |
| interaction_id | string | Interaction identifier linked to spend | I000125 | |
| spend_category | string | Spend category | Meals | Categorical |
| amount_local | numeric | Amount in local currency | 1250.50 | Non-negative |
| currency | string | Currency code | USD / eur | Mostly ISO codes |
| amount_usd | numeric | Amount converted to USD | 1362.05 | |
| payment_date | string (date-like) | Date payment was made | 2024-10-02 | |

## File: `stakeholders.csv`
| Column | Type | Description | Example | Notes |
|---|---|---|---|---|
| stakeholder_id | string | Stakeholder identifier | S00421 | Primary key |
| stakeholder_type | string | Stakeholder type | HCP | HCP or HCO |
| country | string | Stakeholder home country | Germany | |
| specialty | string | Specialty (if HCP) | Oncology | HCOs labeled as `Organization` |
| risk_tier | string | Broad risk tier | Medium | Low/Medium/High |
| onboarding_date | string (date) | Approx onboarding date | 2022-11-03 | Useful for tenure/recency analysis |

## File: `employees.csv`
| Column | Type | Description | Example | Notes |
|---|---|---|---|---|
| employee_id | string | Employee identifier | E00110 | Primary key |
| role | string | Employee role category | MSL | Categorical |
| country | string | Employee home country | Japan | Categorical |
| region | string | Region grouping | APAC | Derived from country |
| tenure_years | numeric | Approx tenure in years | 6.2 | Rounded; used for segmentation |

## Relationship Notes
- `interactions.stakeholder_id` → `stakeholders.stakeholder_id`
- `interactions.employee_id` → `employees.employee_id`
- `spend.interaction_id` → `interactions.interaction_id` (many-to-one)
