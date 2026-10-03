## Data Contract: processed/customers

### Producer
Team / process: Glue ETL job `northstar-dev-transform`

Output is Parquet in `s3://northstar-dev-data-{account-id}/processed/customers/`. Each run overwrites the whole prefix.

### Consumers
- Feature engineering job `northstar-dev-feature-engineer`
- (Future) Direct model training in Lab 3

### Grain
One row per transaction. A customer appears on many rows.

### Schema
| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `transaction_id` | string | No | Unique transaction ID (`TXN-` + 12 characters) |
| `customer_id` | string | No | Customer ID (`CUST-` + 8 digits), repeats across rows |
| `purchase_date` | date | No | Purchase date, parsed from either ISO or MM/DD/YYYY in raw |
| `order_value` | double | No | Order total in USD, nulls filled with the median |
| `num_items` | int | No | Number of items in the order, nulls filled with the median |
| `payment_method` | string | No | `credit_card`, `debit_card`, `gift_card`, `cash`, or `unknown` |
| `channel` | string | No | `online`, `store`, or `unknown` |
| `store_id` | string | No | `STORE-###` for in-store, `ONLINE` for online, or `unknown` |
| `product_category` | string | No | One of 8 product categories, or `unknown` |

### Quality Guarantees
- `customer_id` is never null
- No duplicate `transaction_id` rows (a `customer_id` repeating across rows is expected, not a defect)
- No nulls in any column. Missing numbers get the median and missing strings get `unknown`
- `purchase_date` is a valid ISO 8601 date between 2025-04-01 and 2026-06-30
- `order_value` is between 0 and 10,000, and `num_items` is between 1 and 50

The transform job asserts the first two guarantees and that every `purchase_date` parsed before it writes, so a bad run fails instead of writing bad data.

### SLA
- Data is available in `processed/customers/` within 2 hours of landing in `raw/customers/`

### Versioning
- Schema changes require a new S3 prefix (e.g., `processed/customers/v2/`)
- Breaking changes require consumer notification 5 business days in advance
