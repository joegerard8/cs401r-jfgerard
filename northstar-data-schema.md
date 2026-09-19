# NorthStar Retail — Data Source Schemas
## CS 401R Lab 1 Reference

Use this document to understand the shape of the data before it arrives in Lab 2.

---

## customers.csv

**Volume:** 250,000 rows | **Update cadence:** Daily (CRM sync)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `customer_id` | string | No | Format: `CUST-{8 digits}`. Unique. Primary key across all datasets. |
| `email` | string | No | Hashed in this dataset (SHA-256). Never store raw email in the AI platform. |
| `signup_date` | date | No | ISO 8601 (YYYY-MM-DD). Date customer joined loyalty program. |
| `loyalty_tier` | string | No | One of: `Bronze`, `Silver`, `Gold`, `Platinum`. |
| `loyalty_points` | integer | No | Current points balance. Range: 0–250,000. |
| `preferred_channel` | string | No | One of: `store`, `online`, `both`. Self-reported at signup. |
| `age_band` | string | Yes | One of: `18-24`, `25-34`, `35-44`, `45-54`, `55-64`, `65+`. ~8% null (not collected before 2019). |
| `state` | string | No | Two-letter US/CA state/province code. |
| `zip_code` | string | No | 5-digit US zip or Canadian postal code prefix. |
| `lifetime_spend` | float | No | Total spend since signup, in USD. |
| `churn_label` | integer | No | Binary: 1 = churned (no purchase in the 90 days following the snapshot date), 0 = active. ~22% positive rate in the sample dataset. |
| `churn_date` | date | Yes | Date of last purchase before churn. Null if `churn_label = 0`. |
| `snapshot_date` | date | No | Date this record reflects. All rows use the same snapshot date. |

**Known quality issues:**
- `age_band` is missing for ~8% of customers (pre-2019 signups)
- `zip_code` has ~0.3% invalid codes (data entry errors)
- `lifetime_spend` may be understated for pre-2015 customers (legacy system migration gap)

---

## transactions.parquet

**Volume:** ~4.2M rows | **Cadence:** Nightly batch (POS) + streaming (e-commerce)
**Schema version:** v2.1 (breaking change from v1.x in March 2024 — `store_id` format changed)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `transaction_id` | string | No | Format: `TXN-{12 alphanumeric}`. Unique. |
| `customer_id` | string | No | Foreign key to `customers.csv`. |
| `store_id` | string | No | Format: `STORE-{3 digits}` for in-store (001–400). `ONLINE` for e-commerce. |
| `transaction_date` | datetime | No | UTC. Timezone offset: POS timestamps are local time converted to UTC; e-commerce is UTC native. |
| `transaction_amount` | float | No | Gross transaction value in USD. Includes tax. Excludes shipping for online orders. |
| `net_amount` | float | No | Net after promotions and returns applied. |
| `num_items` | integer | No | Number of line items (not units). |
| `num_units` | integer | No | Total units across all line items. |
| `payment_method` | string | No | One of: `credit_card`, `debit_card`, `loyalty_points`, `gift_card`, `cash`. |
| `promotion_code` | string | Yes | Promotion code applied, if any. Null if no promotion. |
| `promotion_discount` | float | Yes | Dollar discount applied. Null if no promotion. |
| `channel` | string | No | One of: `store`, `online`. |
| `device_type` | string | Yes | One of: `mobile`, `desktop`, `tablet`. Null for in-store transactions. |
| `return_flag` | boolean | No | True if this transaction is a return/refund. |
| `product_categories` | string | No | Pipe-delimited list of product categories in this transaction. e.g., `Apparel|Footwear`. |

**Known quality issues:**
- ~2% of transactions have `customer_id` that does not appear in `customers.csv` (guest checkouts converted to loyalty mid-transaction)
- Pre-March 2024 records have `store_id` in old format (`S{3 digits}`) — requires normalization
- `promotion_discount` has ~0.5% null rate even when `promotion_code` is not null (system bug, known)

---

## clickstream.parquet

**Volume:** ~8.1M rows (90 days) | **Cadence:** Streaming via Kinesis (~90K events/hour peak)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `event_id` | string | No | UUID v4. Unique per event. |
| `customer_id` | string | Yes | Null for anonymous/unauthenticated sessions (~35% of events). |
| `session_id` | string | No | UUID v4. Groups events within a single browsing session. |
| `event_timestamp` | datetime | No | UTC, millisecond precision. |
| `event_type` | string | No | One of: `page_view`, `product_view`, `search`, `add_to_cart`, `remove_from_cart`, `checkout_start`, `checkout_complete`, `checkout_abandon`, `login`, `logout`. |
| `page_name` | string | Yes | Human-readable page name. Null for non-page events. |
| `product_id` | string | Yes | SKU ID from product catalog. Null for non-product events. |
| `search_query` | string | Yes | Raw search query text. Null for non-search events. PII risk: customers sometimes type email addresses. |
| `device_type` | string | No | One of: `mobile`, `desktop`, `tablet`. |
| `referral_source` | string | Yes | One of: `organic`, `email`, `paid_search`, `social`, `direct`, `affiliate`. Null for non-entry events. |
| `cart_value` | float | Yes | Cart value at time of event. Null for non-cart events. |

**Known quality issues:**
- ~35% of events have null `customer_id` (anonymous sessions)
- Bot traffic is not filtered in the raw stream — filter events with session duration < 5 seconds
- `search_query` occasionally contains PII (customer emails) — must be masked before storage

---

## store_events.csv

**Volume:** ~14,400 rows (400 stores × ~36 events/store average over 18 months) | **Cadence:** Manual entry (variable quality)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `store_id` | string | No | Foreign key to transaction `store_id`. |
| `store_name` | string | No | NorthStar store name. |
| `city` | string | No | City name. |
| `state` | string | No | Two-letter state/province code. |
| `event_date` | date | No | ISO 8601. Date of the event or start date for multi-day events. |
| `event_end_date` | date | Yes | End date for multi-day events. Null for single-day events. |
| `event_type` | string | No | One of: `promotion`, `holiday_closure`, `remodel`, `grand_opening`, `clearance_sale`, `inventory_event`, `seasonal_reset`. |
| `event_description` | string | Yes | Free-text description. Quality varies. ~15% null. |
| `revenue_impact_estimate` | float | Yes | Estimated revenue impact (positive or negative). ~40% null (not always recorded). |

**Known quality issues:**
- ~15% of events lack descriptions
- Manual entry leads to inconsistent date formats (some entries use MM/DD/YYYY — requires normalization)
- ~8% of `store_id` values have trailing whitespace (trim required)

---

## product_catalog.json

**Volume:** 12,000 SKUs | **Cadence:** Daily update from PIM system

**Top-level structure:** JSON array of product objects.

```json
[
  {
    "sku_id": "SKU-000001",
    "product_name": "TrailBlazer Waterproof Jacket",
    "category": "Apparel",
    "subcategory": "Outerwear",
    "brand": "NorthStar Own Brand",
    "price": 189.99,
    "sale_price": null,
    "in_stock": true,
    "rating": 4.3,
    "review_count": 847,
    "description": "...",
    "tags": ["waterproof", "hiking", "outdoor", "men"],
    "related_skus": ["SKU-000045", "SKU-000892"],
    "last_updated": "2026-08-15"
  }
]
```

**Product categories (12):** `Apparel`, `Footwear`, `Camping & Hiking`, `Climbing`, `Water Sports`, `Cycling`, `Winter Sports`, `Fitness`, `Travel`, `Electronics`, `Home & Garden`, `Pet Gear`

**Known quality issues:**
- ~3% of SKUs have null `sale_price` when they should have one (PIM sync bug, known)
- `description` field varies from 20 to 800 words — no minimum length enforced
- `related_skus` sometimes references discontinued SKUs not in the active catalog

---

## Data Relationships

```
customers.csv ─────────────────────────────┐
     │                                      │
     │ customer_id                          │
     ├──────────────→ transactions.parquet  │
     │                      │               │
     │                      │ store_id      │
     │                      └──→ store_events.csv
     │
     │ customer_id
     └──────────────→ clickstream.parquet
                            │
                            │ product_id
                            └──→ product_catalog.json
```

**Referential integrity warnings:**
- Not all `customer_id` values in transactions appear in customers (guest checkouts: ~2%)
- Not all `product_id` values in clickstream appear in catalog (discontinued SKUs: ~4%)
- All `store_id` values in transactions should appear in store_events, but coverage is only ~85%
