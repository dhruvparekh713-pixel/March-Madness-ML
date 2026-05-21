# March Madness ML Prediction Model

Predictive bracket model for the Second Annual March Madness Machine Learning Competition at Carnegie Mellon University. Placed **9th out of 40+ competitors** with 90.6% first-round accuracy (29/32 games), including correctly predicting all 8–9 seed matchup upsets.

## Overview

An ensemble machine learning model (Gradient Boosting + Random Forest + Logistic Regression) that predicts NCAA Tournament outcomes using efficiency metrics, tempo analysis, and resume-based features.

### Key Features
- 25-feature enhanced feature set including KenPom/Torvik efficiency metrics, quadrant records, momentum, and a custom "Trapezoid of Excellence" score
- Ensemble of GBM, Random Forest, and Logistic Regression with optimized weighting
- Live stat scraping from Bart Torvik and KenPom
- PDF bracket parsing for Selection Sunday input
- Full pairwise prediction generation (72,390 men's + 71,631 women's matchups)

## Results (2026 Tournament)
- **29/32** Round of 64 games predicted correctly (90.6%)
- All 8–9 seed matchups predicted correctly
- Predicted VCU over UNC upset
- **9th place** out of 40+ competitors

## v1 → v2 Improvements

| Area | v1 | v2 |
|------|----|----|
| Training data | 60 hand-typed games, synthetic stats | 163 real tournament games (2013-2025) with actual O/D splits |
| Team ID mappings | 183/209 entries had wrong IDs | Auto-derived from canonical Kaggle tables, zero conflicts |
| Validation | 5-fold CV on synthetic data | Leave-one-year-out CV + feature importance analysis |
| Feature stats | `adj_o = 100 + adj_em`, `adj_d = 100` for all teams | Real offensive/defensive efficiency per team |
| Diagnostics | None | Feature importance ranking, honest accuracy reporting |

## Usage

```bash
# Install dependencies
pip install scikit-learn numpy pdfplumber requests beautifulsoup4 lxml

# Run the interactive model (v1)
python v1/march_madness_model.py

# Generate competition submissions (v2)
python mmml_competition_v2.py --all

# Scrape live stats first (run locally with internet)
python mmml_competition_v2.py --scrape-only --gender M
python mmml_competition_v2.py --all --bracket bracket.pdf
```

## File Structure

```
v1/                          # Original competition submission
  march_madness_model.py     # Interactive prediction model with Monte Carlo sim
  mmml_competition.py        # v1 submission generator
mmml_competition_v2.py       # Improved model with real training data + diagnostics
data/                        # Scraped stats (gitignored)
```

## Tech Stack
Python, scikit-learn, NumPy, BeautifulSoup4, pdfplumber

## Author
Dhruv Parekh — Carnegie Mellon University, ECE '28
