# Football Analysis Tool

A comprehensive Python-based analysis tool for examining Georgian national football team performance and player statistics.

## Overview

This project provides specialized analysis tools for:
- Georgian player statistics and performance metrics
- Home vs Away performance analysis for the Georgian national team
- Goal scoring patterns and match result analysis

## Data Sources

The tool works with three main CSV files:
1. `goalscorers.csv` - Contains goal scoring data
2. `results.csv` - Contains match results
3. `shootouts.csv` - Contains penalty shootout data

## Installation

### Prerequisites
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn

### Setup
```bash
# Clone the repository
git clone [repository-url]

# Install required packages
pip install -r requirements.txt
```

## Project Structure

```
georgian-football-analysis/
│
├── data/
│   ├── goalscorers.csv
│   ├── results.csv
│   └── shootouts.csv
│
├── src/
│   ├── georgian_player_analyzer.py
│   └── georgian_home_away_analyzer.py
│
└── README.md
```

## Features

### Georgian Player Analyzer
- Individual player statistics tracking
- Goal scoring patterns analysis
- Penalty kick analysis
- Timeline visualizations of goals
- Top scorers rankings

### Georgian Home vs Away Analyzer
- Match result statistics
- Goal distribution analysis
- Tournament performance tracking
- Opponent analysis
- Performance visualizations

## Usage

### Player Analysis
```python
from georgian_player_analyzer import GeorgianPlayerAnalyzer

# Initialize analyzer
analyzer = GeorgianPlayerAnalyzer()

# Load data
analyzer.load_data('Data/InternationalMatches/goalscorers.csv', 'Data/InternationalMatches/results.csv')

# Get player statistics
stats = analyzer.get_georgian_scorers_stats()

# Create visualizations
analyzer.plot_georgian_goals_timeline()
analyzer.plot_top_georgian_scorers()
```

### Home vs Away Analysis
```python
from georgian_home_away_analyzer import GeorgianHomeAwayAnalyzer

# Initialize analyzer
analyzer = GeorgianHomeAwayAnalyzer()

# Load data
analyzer.load_data('Data/InternationalMatches/results.csv', 'Data/InternationalMatches/goalscorers.csv')

# Get performance statistics
stats = analyzer.calculate_overall_statistics()

# Analyze tournament performance
tournament_stats = analyzer.analyze_tournaments()

# Create visualizations
analyzer.plot_performance_comparison()
analyzer.plot_goals_distribution()
```

## Key Features

### Player Statistics
- Total goals scored
- Penalty goals
- Goal timing patterns
- Home vs away goals
- Scoring periods

### Team Performance Metrics
- Win/loss/draw ratios
- Goals scored/conceded
- Tournament performance
- Home vs away comparison
- Opponent analysis

### Visualizations
- Goal timeline plots
- Performance comparison charts
- Goal distribution analyses
- Tournament performance graphs

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data sources and any relevant acknowledgments
- Contributors to the project
- Any third-party libraries or tools used

## Contact

For any queries or suggestions, please open an issue in the repository.
