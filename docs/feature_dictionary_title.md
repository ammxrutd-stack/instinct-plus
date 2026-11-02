# Title feature dictionary

Computed by `src/title/features.py: extract_title_features(title, content_type, subscriber_count=None)`.
Feature version `title-v2`. Every value is a float; booleans are 0/1. Names
are stable: a model card lists the exact set it was trained with.

## Length and shape

| Feature | Definition |
|---|---|
| `char_count` | characters, after stripping surrounding whitespace |
| `word_count` | tokens matching `[A-Za-z0-9'’]+` |
| `avg_word_len`, `longest_word_len` | over those tokens |
| `stopword_ratio` | share of tokens in a small stopword list |
| `unique_word_ratio` | distinct lowercase tokens ÷ tokens |
| `punctuation_count` | count of `! ? . , : ; \| ( ) [ ] " ' - – — …` |

## Numbers and money

| Feature | Definition |
|---|---|
| `has_number`, `number_count` | any digit run (`\d+(?:[.,]\d+)?`) |
| `starts_with_number` | first non-space character is a digit |
| `has_year` | a four-digit 19xx/20xx year |
| `has_money` | a currency symbol before a digit, or "dollars", "usd", "k", "m" after one |
| `is_listicle` | leading number, or "N things/ways/tips/reasons/…" |

## Framing

| Feature | Definition |
|---|---|
| `is_question` | ends with `?` |
| `question_marks` | count of `?` |
| `starts_with_question_word` | opens with why/how/what/when/who/which/where/can/should/is/do/does/did |
| `how_to` | "how to", "how I", "how we", "how you" |
| `is_versus` | "vs", "vs.", "versus" |
| `has_brackets`, `has_parentheses`, `has_pipe`, `has_colon`, `has_dash`, `has_ellipsis`, `has_quotes` | presence of that punctuation (dash = a spaced dash) |
| `first_person` | count of I / my / me / we / our / us / I've / mine |
| `second_person` | count of you / your / you're / yours / you'll / you've |

## Emphasis and tone

| Feature | Definition |
|---|---|
| `caps_word_count`, `caps_word_ratio` | words of 2+ letters entirely in capitals, count and share of alphabetic words |
| `upper_letter_ratio` | capital letters ÷ letters |
| `title_case_ratio` | share of alphabetic words starting with a capital |
| `exclamations` | count of `!` |
| `emoji_count` | characters in the common emoji ranges |
| `power_word_count` | count of words in `POWER_WORDS` (secret, ultimate, best, worst, never, truth, mistake, …) |
| `curiosity_phrase` | count of phrases in `CURIOSITY_PHRASES` ("what happened", "the truth about", "nobody tells you", "I tried", …) |
| `urgency_word_count` | now, today, before, last, finally, new, a current year, … |
| `negation_count` | never, no, not, don't, stop, without, worst, … |
| `superlative_count` | words in `SUPERLATIVE_WORDS` or ending in "est" (5+ letters) |
| `sentiment_compound`, `sentiment_pos`, `sentiment_neg` | VADER sentiment; all zero if `vaderSentiment` is not installed |

## Context

| Feature | Definition |
|---|---|
| `log_subscribers` | log10(subscriber_count); NaN when not supplied, replaced by the training median at scoring time. Given to the model because views-per-subscriber falls with channel size; its contribution is removed from the score and reported separately (`meta.context_points`). |
| `ct_gaming`, `ct_cooking`, `ct_finance`, `ct_education` | one-hot content type; all zero for an unknown category |

## Not included, and why

- Sentence embeddings (`sentence-transformers`): would add topical signal
  but a 384-dimensional vector cannot be explained to a creator as a driver.
  A reasonable extension is a handful of SVD components over embeddings,
  labelled by the titles nearest each axis.
- Named entities (spaCy): the current list of dependencies is already heavy;
  the vocabulary features above cover most of what entities would add.
