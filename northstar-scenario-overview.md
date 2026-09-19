# NorthStar Retail — Case Overview
## CS 401R Reference Document

---

## Company Profile

**NorthStar Retail** is a specialty retailer with 400 stores across North America and a growing direct-to-consumer e-commerce presence. Founded in 1987, NorthStar sells outdoor gear, apparel, and home goods across five major product categories.

| Metric | Value |
|--------|-------|
| Annual Revenue | ~$3.2B |
| Stores | 400 (US, Canada, Mexico) |
| Active Customers (12-month) | ~2.1M |
| E-commerce Share of Revenue | ~28% and growing |
| Average Order Value | $87 (store), $112 (online) |
| Loyalty Program Members | ~1.4M |

---

## The AI Initiative

NorthStar's Chief Data Officer, Maya Chen, has commissioned an AI platform to address a **$128.5M annual problem**: customer churn. Analysis shows that NorthStar loses approximately **18%** of its active customer base each year to inactivity. Each churned customer represents ~**$340** in lost lifetime value.

That headline figure is simply the product of the three numbers above, and you are expected to be able to reproduce it: 2.1M active customers × 18% × $340 = **$128.5M**. Use these figures for every business calculation in the course. Where a lab asks you to size value, cost, or ROI, it should reconcile to this arithmetic.

The CDO has prioritized three interconnected AI systems:

### System 1: Churn Prediction Model
**Problem:** NorthStar cannot identify at-risk customers until after they have already left.
**Solution:** A batch ML model that scores every active customer weekly, predicting **90-day** churn probability — the likelihood that a customer goes silent over the 90 days following the scoring date.
**Business Goal:** Enable proactive outreach before churn occurs, targeting the top 10% highest-risk customers with retention offers.
**Success Metric:** Reduce 12-month churn rate from 18% to 14% within one year of deployment.

### System 2: Offer Generation (LLM / RAG)
**Problem:** Retention offers are generic — the same 20% discount sent to every at-risk customer — resulting in low redemption rates (6%) and unnecessary margin erosion.
**Solution:** A RAG-powered offer generation system that personalizes retention offers based on each customer's purchase history, loyalty tier, and product affinity.
**Business Goal:** Increase offer redemption rate from 6% to 12%; improve offer margin by excluding high-LTV customers from deep discounts.
**Success Metric:** Offer redemption rate and incremental revenue per offer sent.

### System 3: Customer Service Agent
**Problem:** NorthStar's customer service team handles 14,000 contacts per day. 62% are routine inquiries (order status, returns, loyalty points) that do not require human judgment.
**Solution:** An agentic AI system that handles routine contacts autonomously, routing only complex cases to human agents.
**Business Goal:** Automate 50% of routine contacts; reduce average handle time for escalated cases by 30%.
**Success Metric:** Automation rate, CSAT score (Customer Satisfaction Score) for AI-handled contacts, human escalation rate.

---

## Data Architecture (Current State — Pre-AI)

NorthStar currently runs a traditional data warehouse (Snowflake) fed by nightly ETL jobs. The following data sources are available and approved for AI use:

### Transactional Data
- **POS System** (Point of Sale): Every in-store transaction. Batch export nightly at 2 AM ET. Schema: `transaction_id`, `customer_id`, `store_id`, `transaction_date`, `transaction_amount`, `num_items`, `payment_method`, `promotion_code`.
- **E-commerce Platform** (Shopify): Online orders in real-time via webhook. Schema mirrors POS with additions: `session_id`, `device_type`, `referral_source`.

### Customer Data
- **CRM** (Salesforce): Customer master record, loyalty tier, signup date, contact preferences. Updated in near-real-time.
- **Loyalty Platform** (proprietary): Points balance, tier history, offer redemption history. Batch sync every 4 hours.

### Behavioral Data
- **Web/App Analytics** (Segment): Clickstream events from northstar.com and the NorthStar mobile app. Streaming via Kinesis. Volume: ~90,000 events/hour peak.

### Product Data
- **Product Information Management (PIM)**: Full catalog of 12,000 active SKUs with descriptions, categories, pricing, and inventory status. Updated daily.
- **Customer Reviews**: 2.3M product reviews with ratings (1–5 stars). Historical export available.

### Operational Data
- **Store Operations System**: Store-level promotions, hours changes, remodels, closures. Manual entry; quality varies.

---

## The AI Platform (Your Mission)

You are building the AI infrastructure that does not yet exist. NorthStar has approved AWS as the cloud provider. Your platform must:

1. **Ingest** data from the sources above at the appropriate cadence (batch, streaming, event-driven)
2. **Store and transform** data into features suitable for ML training and inference
3. **Train and evaluate** three types of AI systems (traditional ML, LLM/RAG, agentic)
4. **Automate** the model lifecycle from commit to production
5. **Serve** predictions at the appropriate latency (weekly batch for churn, real-time for offers and agent)
6. **Monitor** all systems for drift, performance degradation, and business impact
7. **Measure** the economic and business value of the full platform

---

## Constraints and Requirements

**Regulatory:**
- Customer PII is subject to GDPR (Canadian customers), CCPA (California customers), and NorthStar's internal data retention policy (24-month raw data retention)
- The credit-offer component of the offer generation system is subject to FCRA/ECOA non-discrimination requirements

**Operational:**
- The churn model must produce weekly scores by Monday 6 AM ET (in time for the weekly retention campaign)
- The offer generation system must respond within 2 seconds for real-time offer display
- The customer service agent must maintain 99.5% availability during NorthStar's business hours (8 AM–10 PM in all time zones)

**Budget:**
- Total AI platform budget: $85,000/month (including compute, storage, and third-party services)
- Your personal AWS Free Tier account has up to $200 in credits. Stay within this limit — credits expire in 6 months. See `aws-account-setup.md` for cost controls and shutdown procedures.

---

## NorthStar Organizational Context

The AI team reports to Maya Chen (CDO). Key stakeholders:

| Role                      | Name         | Interest                                                |
| ------------------------- | ------------ | ------------------------------------------------------- |
| Chief Data Officer        | Maya Chen    | Overall AI strategy; board reporting                    |
| VP of Marketing           | David Park   | Churn reduction; offer redemption rates                 |
| VP of Customer Experience | Priya Nair   | CSAT; agent automation rate                             |
| CFO                       | Robert Hess  | Cost per AI interaction; ROI                            |
| Chief Privacy Officer     | Sarah Okafor | Data governance; regulatory compliance                  |
| VP of Technology          | James Wu     | Platform reliability; integration with existing systems |

---

## What You Are NOT Building

To keep scope manageable for a semester course, the following are out of scope:

- Real-time personalization during the browsing session (would require <50ms latency)
- The Snowflake integration (you will use the simulated datasets provided)
- The Salesforce/CRM integration (CRM data is baked into `customers.csv`)
- Production-scale traffic (keep workloads small to preserve your $200 credit balance)
- The CLV model and credit-offer model (referenced in governance chapters but not implemented in labs)
