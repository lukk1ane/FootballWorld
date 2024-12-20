import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class GeorgianPlayerAnalyzer:
    def __init__(self):
        """Initialize the Georgian Player Analyzer"""
        self.goalscorers_df = None
        self.results_df = None
        self.georgian_players = None
        
    def load_data(self, goalscorers_path: str, results_path: str) -> None:
        """
        Load both goalscorers and results data
        """
        try:
            self.goalscorers_df = pd.read_csv(goalscorers_path)
            self.results_df = pd.read_csv(results_path)
            print("Data loaded successfully")
            
            # Filter for Georgian players (when team is Georgia)
            self.georgian_players = self.goalscorers_df[
                self.goalscorers_df['team'] == 'Georgia'
            ].copy()
            
            print(f"Found {len(self.georgian_players)} goals by Georgian players")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def get_georgian_scorers_stats(self) -> pd.DataFrame:
        """
        Get comprehensive statistics for all Georgian goalscorers
        """
        if self.georgian_players is None or len(self.georgian_players) == 0:
            return pd.DataFrame()
            
        stats = (self.georgian_players
            .groupby('scorer')
            .agg({
                'scorer': 'count',  # Total goals
                'penalty': 'sum',   # Penalty goals
                'own_goal': 'sum',  # Own goals
                'minute': ['mean', 'std', 'min', 'max'],  # Timing stats
                'date': ['min', 'max']  # First and last goals
            }))
        
        # Rename columns for clarity
        stats.columns = [
            'total_goals', 'penalties', 'own_goals',
            'avg_minute', 'std_minute', 'earliest_goal', 'latest_goal',
            'first_goal_date', 'last_goal_date'
        ]
        
        # Calculate non-penalty goals
        stats['non_penalty_goals'] = stats['total_goals'] - stats['penalties'].fillna(0)
        
        # Convert dates to datetime
        stats['first_goal_date'] = pd.to_datetime(stats['first_goal_date'])
        stats['last_goal_date'] = pd.to_datetime(stats['last_goal_date'])
        
        # Calculate scoring period in days
        stats['scoring_period_days'] = (
            stats['last_goal_date'] - stats['first_goal_date']
        ).dt.days
        
        return stats.sort_values('total_goals', ascending=False)

    def analyze_goal_patterns(self) -> dict:
        """
        Analyze patterns in Georgian goals
        """
        if self.georgian_players is None or len(self.georgian_players) == 0:
            return {}
            
        patterns = {
            'total_goals': len(self.georgian_players),
            'unique_scorers': self.georgian_players['scorer'].nunique(),
            'penalties': self.georgian_players['penalty'].sum(),
            'own_goals': self.georgian_players['own_goal'].sum(),
            'avg_minute': self.georgian_players['minute'].mean(),
            'home_goals': len(self.georgian_players[
                self.georgian_players['team'] == self.georgian_players['home_team']
            ]),
            'away_goals': len(self.georgian_players[
                self.georgian_players['team'] != self.georgian_players['home_team']
            ])
        }
        
        # Add time period analysis
        patterns['time_periods'] = {
            'first_half': len(self.georgian_players[self.georgian_players['minute'] <= 45]),
            'second_half': len(self.georgian_players[self.georgian_players['minute'] > 45])
        }
        
        return patterns

    def plot_georgian_goals_timeline(self, save_path: str = None) -> None:
        """
        Plot timeline of Georgian goals
        """
        if self.georgian_players is None or len(self.georgian_players) == 0:
            print("No data available for Georgian players")
            return
            
        plt.figure(figsize=(15, 6))
        
        # Convert date to datetime if needed
        self.georgian_players['date'] = pd.to_datetime(self.georgian_players['date'])
        
        # Create the timeline plot
        plt.plot(self.georgian_players['date'], self.georgian_players['minute'], 'o')
        
        plt.title('Timeline of Georgian Goals')
        plt.xlabel('Date')
        plt.ylabel('Minute of Goal')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_top_georgian_scorers(self, top_n: int = 5, save_path: str = None) -> None:
        """
        Plot bar chart of top Georgian scorers
        """
        if self.georgian_players is None or len(self.georgian_players) == 0:
            print("No data available for Georgian players")
            return
            
        # Get goal counts by player
        scorer_counts = (self.georgian_players['scorer']
            .value_counts()
            .head(top_n))
            
        plt.figure(figsize=(12, 6))
        scorer_counts.plot(kind='bar')
        plt.title(f'Top {top_n} Georgian Goalscorers')
        plt.xlabel('Player')
        plt.ylabel('Number of Goals')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def get_matches_with_georgian_goals(self) -> pd.DataFrame:
        """
        Get details of all matches where Georgian players scored
        """
        if self.georgian_players is None or len(self.georgian_players) == 0:
            return pd.DataFrame()
            
        # Get unique matches where Georgians scored
        matches = self.georgian_players[['date', 'home_team', 'away_team']].drop_duplicates()
        
        # Get scorers for each match
        match_scorers = (self.georgian_players
            .groupby(['date', 'home_team', 'away_team'])['scorer']
            .agg(list)
            .reset_index()
        )
        
        return match_scorers

def main():
    # Initialize analyzer
    analyzer = GeorgianPlayerAnalyzer()
    
    try:
        # Load data
        analyzer.load_data('Data/InternationalMatches/goalscorers.csv', 'Data/InternationalMatches/results.csv')
        
        # Get Georgian player statistics
        stats = analyzer.get_georgian_scorers_stats()
        print("\nGeorgian Goalscorers Statistics:")
        print(stats)
        
        # Analyze goal patterns
        patterns = analyzer.analyze_goal_patterns()
        print("\nGeorgian Goal Patterns:")
        print(patterns)
        
        # Create visualizations
        analyzer.plot_georgian_goals_timeline()
        analyzer.plot_top_georgian_scorers()
        
        # Get matches with Georgian goals
        matches = analyzer.get_matches_with_georgian_goals()
        print("\nMatches with Georgian Goals:")
        print(matches)
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()