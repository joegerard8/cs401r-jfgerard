## Lab 3 Model Design

**Track: A (churn model) + B (offer generation with RAG)**

The churn results are in `lab3-evaluation-report.md`. The RAG results, examples and failures are in `lab3-evidence/track-b-examples.md`.

### 1. Why gradient boosting for the churn model

The training set is 6,999 customers, only 1,540 of them churners, with 11 tabular features. The hard part is the churners who were still buying right before T. Recency says they are safe, and they mostly show up as a recent buyer with a low order value ($106 per order vs $153 for customers who stayed). That is an interaction, and recency also acts more like a step than a straight line.

Logistic regression is additive, so I would have to build those bins and interactions by hand. XGBoost finds them on its own and does not care that the features are in different units. A neural network does not fit here. There is no sequence or text data, 1,540 positives is not much to train on, and it would be the hardest to explain. The retention team needs to understand the model, so I kept the trees at depth 3, which means each path only combines three features.

I would switch to logistic regression if a regularized version with binned recency scored within the CI of XGBoost, since the coefficients are easier for the business to read. I did not run that comparison, but it is the first thing I would add.

### 2. What the features bought

The recency-only baseline scored 0.7223 AUC. The full model scored 0.7827. The lift is +0.0605, with a 95% CI of [0.0422, 0.0778] from a paired bootstrap, so it is real, but it is not huge.

Most of the lift comes from the spend features. Removing average order value, 90-day spend and lifetime value cost 0.031 AUC, CI [-0.046, -0.015]. That was the only ablation that was significant. Tenure, basket size, channel mix and category diversity were decorative. Dropping all four only cost 0.005, which is inside the noise. I was surprised by category diversity, since it was built to catch these churners, but its signal mostly overlaps with purchase frequency. `churn_risk_score` added nothing because it is just a transform of recency.

The lift also mostly helps further down the ranking. The top 10% is mostly customers who stopped buying a long time ago, and recency already finds them. So recall@10% barely changed across the experiments.

### 3. Why RAG instead of fine-tuning

The offers have to get policy facts right, like tier thresholds, benefits and return windows. Fine-tuning would put those facts in the model weights, and the policies change (POL-RET-004 replaced POL-RET-003, and the loyalty terms can change with 60 days notice), so every change would mean retraining. With RAG I just re-embed the 66 chunks, which takes about 20 seconds, and each offer records which chunks it used. There is also no training set of good offers. Writing one by hand is how the planted errors got into Template 3.

I also could have skipped retrieval and put the whole corpus (about 6K tokens) in every prompt. My prompts are around 2K tokens, so that would be about 3.5 times the input cost, with irrelevant rules like Mexico shipping in every prompt.

Lab 2's `loyalty_tier` uses lifetime spend, but the policy uses trailing 12-month spend. About 19% of customers are within 10% of a threshold, where the two can disagree. For those customers my pipeline only quotes the lower tier's benefits, since those are true either way, and it never says how far they are from the next tier. In production I would read the tier from the loyalty system.

The main production risk is sending an invalid offer, like a benefit the customer's tier does not have. Faithfulness does not catch this well. One version of my pipeline said "you're just $15 away from Gold", and the judge marked it supported because the number came from my own profile context. I would add a check before sending that every number and tier in an offer appears in a retrieved chunk or the profile.

### 4. What I would do with 10x the time or data

For the churn model, I would test it the way it will be used: build the features at several T dates, train on the earlier ones and test on the later ones, instead of one random split. I would also early stop on a separate validation set and rank customers by expected value at risk (churn probability times lifetime value), since only 10 of the 300 flagged customers are Gold or Platinum right now.

For the offers, I would have the model pick benefit IDs from a table built from the loyalty terms and let code fill in the numbers. Then it could not make up a number, and I could check offers exactly instead of relying on an LLM judge. I would also build a test set of around 200 profiles covering every tier and channel.
