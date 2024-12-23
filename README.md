# Football Analytics Platform

A machine learning platform analyzing international football and Georgian national team performance, focusing on goal prediction and team analytics.

## Overview

### General Architecture
The platform consists of four specialized analyzers, each handling different aspects of football analytics:

1. **International Goal Pattern Analysis** (`GoalAnalyzer`)
   - Predicts late goals (75+ minute) across international matches
   - Features: temporal analysis, home/away impact, penalty patterns
   - Uses Random Forest and Logistic Regression with balanced class weights 
   - Advanced feature engineering for cross-team pattern detection
   - Implements automated data cleaning and ML pipeline

2. **Georgian Player Analysis** (`GeorgianPlayerAnalyzer`)
   - Tracks Georgian players' performance metrics
   - Analyzes scoring patterns and match contributions
   - Generates comprehensive player statistics

3. **Georgian Home/Away Analysis** (`GeorgianHomeAwayAnalyzer`)
   - Compares team performance in different venues
   - Analyzes tournament-specific patterns
   - Tracks opponent-level statistics

4. **Match Prediction** (`MatchPredictor`)
   - Predicts match outcomes using ensemble methods
   - Implements cross-validation and hyperparameter tuning
   - Generates feature importance analysis

## Data Requirements

Three CSV datasets required:

1. **results.csv** (~4610 records)
   - Match results and metadata
   - Schema: date, home_team, away_team, home_score, away_score, tournament, city, country, neutral
   - Covers international matches with detailed location information

2. **goalscorers.csv** (~4759 records)
   - Individual goal events
   - Schema: date, home_team, away_team, team, scorer, minute, own_goal, penalty
   - Records every goal with timing and type

3. **shootouts.csv** (~89 records)
   - Penalty shootout details
   - Schema: date, home_team, away_team, winner, first_shooter
   - Tracks shootout outcomes and first shooters

## Installation & Setup

```bash
# Clone repository
git clone https://github.com/yourusername/football-analytics.git

# Install dependencies
pip install -r requirements.txt

# Set up data directory
mkdir -p Data/InternationalMatches
```

## Usage Examples

```python
# International Goal Analysis
from goal_analyzer import GoalAnalyzer
analyzer = GoalAnalyzer()
analyzer.load_and_clean_data()
analyzer.prepare_ml_features()
rf_model, lr_model = analyzer.train_models()

# Georgian Player Analysis
from georgian_player_analyzer import GeorgianPlayerAnalyzer
georgian_analyzer = GeorgianPlayerAnalyzer()
georgian_analyzer.load_data('goalscorers.csv', 'results.csv')
stats = georgian_analyzer.get_georgian_scorers_stats()

# Match Prediction
from match_predictor import MatchPredictor
predictor = MatchPredictor()
predictor.load_and_process_data()
results = predictor.train_models()
```

## Project Structure

```
football-analytics/
├── Data/
│   └── InternationalMatches/
│       ├── goalscorers.csv
│       ├── results.csv
│       └── shootouts.csv
├── Visualizations/
│   └── MatricesForInternationalGoalScorers/
├── goal_analyzer.py
├── georgian_player_analyzer.py
├── georgian_home_away_analyzer.py
└── match_predictor.py
```

## Dependencies
- pandas>=1.3.0
- numpy>=1.20.0
- scikit-learn>=0.24.0
- matplotlib>=3.4.0 
- seaborn>=0.11.0
- xgboost>=1.4.0

## DataSource
https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017/data?select=goalscorers.csv
