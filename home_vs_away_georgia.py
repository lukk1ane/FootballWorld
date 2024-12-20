import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple

class GeorgianHomeAwayAnalyzer:
    def __init__(self):
        """Initialize the Georgian Home vs Away Analyzer"""
        self.results_df = None
        self.goalscorers_df = None
        self.georgia_home_matches = None
        self.georgia_away_matches = None
        
    def load_data(self, results_path: str, goalscorers_path: str) -> None:
        """
        Load both results and goalscorers data
        """
        try:
            self.results_df = pd.read_csv(results_path)
            self.goalscorers_df = pd.read_csv(goalscorers_path)
            
            # Filter matches for Georgia
            self.georgia_home_matches = self.results_df[
                self.results_df['home_team'] == 'Georgia'
            ].copy()
            
            self.georgia_away_matches = self.results_df[
                self.results_df['away_team'] == 'Georgia'
            ].copy()
            
            print(f"Found {len(self.georgia_home_matches)} home matches and "
                  f"{len(self.georgia_away_matches)} away matches for Georgia")
                  
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def calculate_overall_statistics(self) -> Dict:
        """
        Calculate comprehensive statistics for Georgia's home and away performances
        """
        if self.georgia_home_matches is None or self.georgia_away_matches is None:
            raise ValueError("Data not loaded. Please load data first.")
            
        stats = {
            'home': {
                'matches_played': len(self.georgia_home_matches),
                'wins': len(self.georgia_home_matches[
                    self.georgia_home_matches['home_score'] > 
                    self.georgia_home_matches['away_score']
                ]),
                'draws': len(self.georgia_home_matches[
                    self.georgia_home_matches['home_score'] == 
                    self.georgia_home_matches['away_score']
                ]),
                'losses': len(self.georgia_home_matches[
                    self.georgia_home_matches['home_score'] < 
                    self.georgia_home_matches['away_score']
                ]),
                'goals_scored': self.georgia_home_matches['home_score'].sum(),
                'goals_conceded': self.georgia_home_matches['away_score'].sum()
            },
            'away': {
                'matches_played': len(self.georgia_away_matches),
                'wins': len(self.georgia_away_matches[
                    self.georgia_away_matches['away_score'] > 
                    self.georgia_away_matches['home_score']
                ]),
                'draws': len(self.georgia_away_matches[
                    self.georgia_away_matches['away_score'] == 
                    self.georgia_away_matches['home_score']
                ]),
                'losses': len(self.georgia_away_matches[
                    self.georgia_away_matches['away_score'] < 
                    self.georgia_away_matches['home_score']
                ]),
                'goals_scored': self.georgia_away_matches['away_score'].sum(),
                'goals_conceded': self.georgia_away_matches['home_score'].sum()
            }
        }
        
        # Calculate additional metrics
        for location in ['home', 'away']:
            stats[location].update({
                'win_percentage': (stats[location]['wins'] / 
                                 stats[location]['matches_played'] * 100),
                'draw_percentage': (stats[location]['draws'] / 
                                  stats[location]['matches_played'] * 100),
                'loss_percentage': (stats[location]['losses'] / 
                                  stats[location]['matches_played'] * 100),
                'avg_goals_scored': (stats[location]['goals_scored'] / 
                                   stats[location]['matches_played']),
                'avg_goals_conceded': (stats[location]['goals_conceded'] / 
                                     stats[location]['matches_played']),
                'goal_difference': (stats[location]['goals_scored'] - 
                                  stats[location]['goals_conceded'])
            })
            
        return stats

    def analyze_tournaments(self) -> pd.DataFrame:
        """
        Analyze Georgia's performance in different tournaments
        """
        def get_tournament_stats(matches: pd.DataFrame, is_home: bool) -> pd.DataFrame:
            score_col = 'home_score' if is_home else 'away_score'
            opponent_score_col = 'away_score' if is_home else 'home_score'
            
            return matches.groupby('tournament').agg({
                'tournament': 'count',  # number of matches
                score_col: ['sum', 'mean'],  # goals scored
                opponent_score_col: ['sum', 'mean']  # goals conceded
            })
        
        home_stats = get_tournament_stats(self.georgia_home_matches, True)
        away_stats = get_tournament_stats(self.georgia_away_matches, False)
        
        return pd.concat([home_stats, away_stats], 
                        keys=['Home', 'Away'], 
                        axis=0)

    def get_goal_scorers_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyze Georgian goalscorers in home vs away matches
        """
        georgian_goals = self.goalscorers_df[
            self.goalscorers_df['team'] == 'Georgia'
        ]
        
        home_scorers = georgian_goals[
            georgian_goals['home_team'] == 'Georgia'
        ]['scorer'].value_counts()
        
        away_scorers = georgian_goals[
            georgian_goals['away_team'] == 'Georgia'
        ]['scorer'].value_counts()
        
        return home_scorers, away_scorers

    def plot_performance_comparison(self, save_path: str = None) -> None:
        """
        Plot comparison of home vs away performance metrics
        """
        stats = self.calculate_overall_statistics()
        
        metrics = ['win_percentage', 'draw_percentage', 'loss_percentage']
        values_home = [stats['home'][metric] for metric in metrics]
        values_away = [stats['away'][metric] for metric in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, values_home, width, label='Home')
        plt.bar(x + width/2, values_away, width, label='Away')
        
        plt.xlabel('Metrics')
        plt.ylabel('Percentage')
        plt.title('Georgia\'s Home vs Away Performance')
        plt.xticks(x, [m.replace('_', ' ').title() for m in metrics])
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_goals_distribution(self, save_path: str = None) -> None:
        """
        Plot distribution of goals scored and conceded, home vs away
        """
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting
        home_scored = self.georgia_home_matches['home_score']
        home_conceded = self.georgia_home_matches['away_score']
        away_scored = self.georgia_away_matches['away_score']
        away_conceded = self.georgia_away_matches['home_score']
        
        plt.subplot(1, 2, 1)
        plt.title('Goals Scored Distribution')
        plt.boxplot([home_scored, away_scored], labels=['Home', 'Away'])
        plt.ylabel('Number of Goals')
        
        plt.subplot(1, 2, 2)
        plt.title('Goals Conceded Distribution')
        plt.boxplot([home_conceded, away_conceded], labels=['Home', 'Away'])
        plt.ylabel('Number of Goals')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def analyze_opponent_levels(self) -> pd.DataFrame:
        """
        Analyze performance against different opponents based on FIFA rankings
        """
        # This is a placeholder for FIFA rankings - you would need to add actual rankings
        def get_performance_vs_opponent(matches: pd.DataFrame, is_home: bool) -> pd.DataFrame:
            team_col = 'away_team' if is_home else 'home_team'
            score_col = 'home_score' if is_home else 'away_score'
            opp_score_col = 'away_score' if is_home else 'home_score'
            
            return matches.groupby(team_col).agg({
                team_col: 'count',  # number of matches
                score_col: ['sum', 'mean'],  # goals scored
                opp_score_col: ['sum', 'mean']  # goals conceded
            })
        
        home_analysis = get_performance_vs_opponent(self.georgia_home_matches, True)
        away_analysis = get_performance_vs_opponent(self.georgia_away_matches, False)
        
        return pd.concat([home_analysis, away_analysis], 
                        keys=['Home', 'Away'], 
                        axis=0)

def main():
    # Initialize analyzer
    analyzer = GeorgianHomeAwayAnalyzer()
    
    try:
        # Load data
        analyzer.load_data('Data/InternationalMatches/results.csv', 'Data/InternationalMatches/goalscorers.csv')
        
        # Get overall statistics
        stats = analyzer.calculate_overall_statistics()
        print("\nOverall Statistics:")
        print(stats)
        
        # Analyze tournaments
        tournament_stats = analyzer.analyze_tournaments()
        print("\nTournament Analysis:")
        print(tournament_stats)
        
        # Get goalscorers analysis
        home_scorers, away_scorers = analyzer.get_goal_scorers_analysis()
        print("\nTop Home Scorers:")
        print(home_scorers.head())
        print("\nTop Away Scorers:")
        print(away_scorers.head())
        
        # Create visualizations
        analyzer.plot_performance_comparison()
        analyzer.plot_goals_distribution()
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()