# Phase 1 — Data Exploration

## Class balance

Queries per article: Mean 9.0, Min 9, Max 9, Std 0.00
Total queries: 216 (Expected: 216)
Unique articles: 24 (Expected: 24)
Dataset is perfectly balanced.

## Query length

### Train Queries
- **char_count**: mean 39.7, median 39.0, min 20, max 63, p90 52.0
- **word_count**: mean 7.5, median 7.0, min 3, max 13, p90 10.0
- **token_count**: mean 7.5, median 7.0, min 3, max 13, p90 10.0
### Test Queries
- **char_count**: mean 41.1, median 40.5, min 22, max 65, p90 53.0
- **word_count**: mean 7.9, median 8.0, min 3, max 15, p90 11.0
- **token_count**: mean 7.9, median 8.0, min 3, max 15, p90 11.0


## Filler/noise patterns

Top 1-token prefixes: 'how'(29), 'need'(27), 'hi'(26), 'help'(23), 'question'(23)
Top 2-token prefixes: 'how do'(29), 'need help'(26), 'can you'(21), 'hi give'(4), 'bank refused'(3)
Top 3-token prefixes: 'how do i'(29), 'can you tell'(21), 'bank refused the'(3), 'get money faster'(3), 'hi cost of'(2)
Top 1-token suffixes: 'asap'(32), 'shop'(29), 'account'(25), 'please'(25), 'chargeback'(4)
Top 2-token suffixes: 'my shop'(29), 'my account'(21), 'a chargeback'(4), 'per transaction'(3), 'card asap'(3)
Top 3-token suffixes: 'for my shop'(29), 'on my account'(21), 'take per transaction'(3), 'decline code mean'(3), 'without real money'(3)


## Article-unknown vocabulary

Unknown token ratio: mean 0.403, median 0.400, max 0.857
Most frequent unknown tokens: 'help'(49), 'do'(37), 'asap'(32), 'shop'(29), 'hi'(26), 'please'(25), 'question'(23), 'tell'(21), 'me'(21), 'get'(12)


## Query/article vocabulary overlap

Token overlap ratio: mean 0.321, median 0.333, max 0.833
Unique token overlap ratio: mean 0.318, median 0.317, max 0.833

## Vocabulary-gap examples

**Query**: hi, dunning settings for subscriptions (0.00 overlap)
**Title**: Retrying failed subscription payments

**Query**: dunning settings for subscriptions (0.00 overlap)
**Title**: Retrying failed subscription payments

**Query**: why clsoe business acocunt (0.00 overlap)
**Title**: Closing your account

**Query**: need help: get my 1099 form (0.00 overlap)
**Title**: Downloading invoices and tax documents

**Query**: why get my 1099 form? (0.00 overlap)
**Title**: Downloading invoices and tax documents

**Query**: stop fake orders with stolen cards (0.00 overlap)
**Title**: Preventing fraudulent charges

**Query**: how do i switch where deposits go (0.00 overlap)
**Title**: Change the bank account for payouts

**Query**: how do i switch where deposits go? (0.00 overlap)
**Title**: Change the bank account for payouts

**Query**: get money faster same day on my account (0.00 overlap)
**Title**: Instant payouts to a debit card

**Query**: question: webhook endpoint not receiving events (0.00 overlap)
**Title**: Setting up webhooks

## Train vs test comparison

Train - Avg words: 7.5, Avg unknown ratio: 0.403
Test  - Avg words: 7.9, Avg unknown ratio: 0.377

## Article similarity/difficulty observations

Category distribution: {'Payouts': 4, 'Payments': 4, 'Account': 4, 'Disputes': 3, 'Subscriptions': 3, 'Refunds': 2, 'Billing': 2, 'Developers': 2}
Title length: mean 28.6
Body tokens: mean 30.5

## Key findings
(Generated from analysis)