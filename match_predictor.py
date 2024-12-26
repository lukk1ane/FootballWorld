import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import warnings
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


class MatchPredictor:
    def __init__(self):
        """Initialization"""
        self.results_df = None
        self.goalscorers_df = None
        self.shootouts_df = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.feature_names = None
        self.label_encoders = {}

        # output directory for plots
        self.output_dir = 'match_predictions'
        for dir_name in ['matrices', 'metrics', 'feature_importance']:
            os.makedirs(os.path.join(self.output_dir, dir_name), exist_ok=True)

    def load_and_process_data(self):
        """Loading and preprocessing all data"""
        try:
            # Load datasets
            self.results_df = pd.read_csv('Data/InternationalMatches/results.csv')
            self.goalscorers_df = pd.read_csv('Data/InternationalMatches/goalscorers.csv')
            self.shootouts_df = pd.read_csv('Data/InternationalMatches/shootouts.csv')

            # Converting dates from string to date time
            for df in [self.results_df, self.goalscorers_df, self.shootouts_df]:
                df['date'] = pd.to_datetime(df['date'])

            # target variable: outcome/result we want our model to predict (whether the home team wins)
            self.results_df['home_win'] = (self.results_df['home_score'] > self.results_df['away_score']).astype(int)

            self._process_shootouts()

            self._process_goal_patterns()

            # calculating team statistics (team performance metrics)
            self._add_team_statistics()

            self._process_tournament_features()

            print("\nData processing completed successfully!")
            print(f"Final dataset shape: {self.results_df.shape}")

            return self.results_df

        except Exception as e:
            print(f"Error in data processing: {str(e)}")
            raise

    def _process_shootouts(self):
        """Process shootout information"""
        # processes data related to shootouts from two datasets—results_df and shootouts_df—and integrates
        # this information into the main dataset, enhancing the dataset with relevant shootout features

        # Create a mask for matches that went to shootouts
        # merging self.results_df with self.shootouts_df
        shootout_matches = pd.merge(
            self.results_df,
            self.shootouts_df[['date', 'home_team', 'away_team', 'winner', 'first_shooter']],
            on=['date', 'home_team', 'away_team'],
            how='left'
        )

        # adding features
        self.results_df['went_to_shootout'] = shootout_matches['winner'].notna().astype(int)
        self.results_df['won_shootout'] = (shootout_matches['winner'] == shootout_matches['home_team']).astype(int)
        self.results_df['shootout_first'] = (shootout_matches['first_shooter'] == shootout_matches['home_team']).astype(
            int)

    def _process_goal_patterns(self):
        """Process goal scoring patterns with enhanced error handling"""
        try:
            # grouping the goalscorers_df datarframe by date, home_team, and away_team
            # and for each match calculating count, mean, sum, and std
            match_goals = self.goalscorers_df.groupby(['date', 'home_team', 'away_team']).agg({
                'minute': ['count', 'mean', 'std'],
                'penalty': 'sum',
                'own_goal': 'sum'
            }).reset_index()

            # flattening
            match_goals.columns = ['date', 'home_team', 'away_team',
                                   'goals_in_match', 'avg_goal_minute',
                                   'goal_minute_std', 'penalties', 'own_goals']

            # Convert date to datetime if it isn't already
            match_goals['date'] = pd.to_datetime(match_goals['date'])
            self.results_df['date'] = pd.to_datetime(self.results_df['date'])

            # goal-scoring statistics are merged back into the main results_df
            self.results_df = pd.merge(
                self.results_df,
                match_goals,
                on=['date', 'home_team', 'away_team'],
                how='left'
            )

            # using appropriate default values to fill missing values
            numeric_columns = ['goals_in_match', 'penalties', 'own_goals']
            for col in numeric_columns:
                self.results_df[col] = self.results_df[col].fillna(0)

            # fill goal timing stats with median values
            timing_columns = ['avg_goal_minute', 'goal_minute_std']
            for col in timing_columns:
                median_value = self.results_df[col].median()
                if pd.isna(median_value):
                    median_value = 45  # default to middle of game if no data
                self.results_df[col] = self.results_df[col].fillna(median_value)

            # verifying no NaN values remain
            nan_check = self.results_df[numeric_columns + timing_columns].isna().sum()
            if nan_check.any():
                print("\nWarning: Some NaN values remain after goal pattern processing:")
                print(nan_check[nan_check > 0])

        except Exception as e:
            print(f"Error in goal pattern processing: {str(e)}")
            default_columns = {
                'goals_in_match': 0,
                'avg_goal_minute': 45,
                'goal_minute_std': 0,
                'penalties': 0,
                'own_goals': 0
            }
            for col, default_value in default_columns.items():
                if col not in self.results_df.columns:
                    self.results_df[col] = default_value

    def _add_team_statistics(self):
        """Calculating basic team statistics"""

        # Calculate rolling stats for each team
        for team_type in ['home_team', 'away_team']:
            # creating a copy of the dataframe sorted by date, so the calculations are done in chronological order
            sorted_df = self.results_df.sort_values('date')

            # creating a dictionary of team statistics "recent_performance"
            # and adding statistics for both teams to the results dataframe
            recent_performance = {}
            for team in sorted_df[team_type].unique():
                team_matches = sorted_df[sorted_df[team_type] == team]
                # calculating the average win rate for the last 5 matches.
                recent_performance[team] = team_matches['home_win'].rolling(
                    window=5,
                    min_periods=1
                ).mean()

            # store results
            self.results_df[f'{team_type}_recent_form'] = self.results_df[team_type].map(
                lambda x: recent_performance.get(x, pd.Series()).iloc[-1] \
                    if x in recent_performance else 0
            )

            # creating a dictionary of team statistics and adding
            # statistics for both teams to the results dataframe
            team_stats = {}
            for team in self.results_df[team_type].unique():
                team_matches = self.results_df[self.results_df[team_type] == team]
                team_stats[team] = {
                    'win_rate': team_matches['home_win'].mean(),
                    'avg_goals': team_matches['goals_in_match'].mean() if 'goals_in_match' in team_matches else 0,
                    'std_goals': team_matches['goals_in_match'].std() if 'goals_in_match' in team_matches else 0,
                    'shootout_rate': team_matches['went_to_shootout'].mean() if 'went_to_shootout' in team_matches else 0
                }

            self.results_df[f'{team_type}_win_rate'] = self.results_df[team_type].map(
                lambda x: team_stats[x]['win_rate'] if x in team_stats else 0
            )
            self.results_df[f'{team_type}_avg_goals'] = self.results_df[team_type].map(
                lambda x: team_stats[x]['avg_goals'] if x in team_stats else 0
            )
            self.results_df[f'{team_type}_std_goals'] = self.results_df[team_type].map(
                lambda x: team_stats[x]['std_goals'] if x in team_stats else 0
            )
            self.results_df[f'{team_type}_shootout_rate'] = self.results_df[team_type].map(
                lambda x: team_stats[x]['shootout_rate'] if x in team_stats else 0
            )

    def _process_tournament_features(self):
        """Process tournament-related features"""

        self.results_df['is_friendly'] = (self.results_df['tournament'] == 'Friendly').astype(int)

        # tournament - average goals per match
        tournament_importance = self.results_df.groupby('tournament').agg({
            'goals_in_match': 'mean'
        }).reset_index()

        tournament_importance['tournament_importance'] = \
            (tournament_importance['goals_in_match'] - tournament_importance['goals_in_match'].min()) / \
            (tournament_importance['goals_in_match'].max() - tournament_importance['goals_in_match'].min())

        self.results_df = pd.merge(
            self.results_df,
            tournament_importance[['tournament', 'tournament_importance']],
            on='tournament',
            how='left'
        )

    def prepare_features(self):
        """Preparing dataset for model training, with proper handling of missing values"""

        try:
            # Select features based on actual columns
            # team_recent_form - recent performance
            # win_rate - win rate of a team
            # avg_goals - average goals of a team
            # std_goals - standard deviation of goals scored
            # shootout_rate - proportion of matches involving the opposite team that went to a shootout
            # tournament_importance - a normalized score representing the importance of the tournament
            numeric_features = [
                'home_team_recent_form', 'away_team_recent_form',
                'home_team_win_rate', 'away_team_win_rate',
                'home_team_avg_goals', 'away_team_avg_goals',
                'home_team_std_goals', 'away_team_std_goals',
                'home_team_shootout_rate', 'away_team_shootout_rate',
                'tournament_importance'
            ]

            binary_features = [
                'neutral', 'is_friendly', 'went_to_shootout'
            ]

            # Create feature matrix
            X = pd.concat([
                self.results_df[numeric_features],
                self.results_df[binary_features]
            ], axis=1)

            # filling in NaN values
            for feature in numeric_features:
                X[feature] = X[feature].fillna(X[feature].mean())
            for feature in binary_features:
                X[feature] = X[feature].fillna(0)

            nan_counts = X.isna().sum()
            if nan_counts.any():
                print("\nWarning: Remaining NaN values after initial filling:")
                print(nan_counts[nan_counts > 0])

            X = X.fillna(0)

            # Store feature names
            self.feature_names = X.columns.tolist()

            # Scaling
            # computes the mean and standard deviation based
            # on the train set and scales the features accordingly
            scaler = StandardScaler()
            X[numeric_features] = scaler.fit_transform(X[numeric_features])

            # target variable that the model will predict
            y = self.results_df['home_win']

            # splitting the data
            split_date = self.results_df['date'].sort_values().iloc[int(len(self.results_df) * 0.8)]
            # the model is trained on older data and tested on newer data
            train_mask = self.results_df['date'] <= split_date
            self.X_train = X[train_mask]
            self.X_test = X[~train_mask]
            self.y_train = y[train_mask]
            self.y_test = y[~train_mask]

            print("\nFeature preparation completed!")
            print(f"Training set shape: {self.X_train.shape}")
            print(f"Testing set shape: {self.X_test.shape}")

            return self.X_train, self.X_test, self.y_train, self.y_test

        except Exception as e:
            print(f"Error in feature preparation: {str(e)}")
            raise

    def train_models(self):
        """Train and evaluate models with cross-validation and hyperparameter tuning"""
        try:
            # Defining models, by creating dictionaries:
            #                 model: the actual classifier instance
            #                 params: a dictionary of hyperparameters to tune with their possible values
            models = {
                'RandomForest': {
                    'model': RandomForestClassifier(random_state=42),
                    'params': {
                        'n_estimators': [100, 200], # number of trees in the forest
                        'max_depth': [10, 20], # maximum depth of each tree
                        'min_samples_split': [2, 5] # minimum samples required to split a node
                    }
                },
                'GradientBoosting': {
                    'model': GradientBoostingClassifier(random_state=42),
                    'params': {
                        'n_estimators': [100, 200], # number of boosting stages
                        'learning_rate': [0.01, 0.1], # how much each tree contributes
                        'max_depth': [3, 5] # maximum depth of each tree
                    }
                },
                'XGBoost': {
                    'model': XGBClassifier(random_state=42),
                    'params': {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.01, 0.1],
                        'max_depth': [3, 5]
                    }
                }
            }

            results = {}

            for name, config in models.items():
                print(f"\nTraining {name}...")

                # Cross-validation
                # Splits training data into 5 folds
                # train on 4, validate on 1
                cv_scores = cross_val_score(
                    config['model'],
                    self.X_train,
                    self.y_train,
                    cv=5,
                    scoring='accuracy'
                )
                print(f"Cross-validation scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

                # Grid search
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1
                )
                grid_search.fit(self.X_train, self.y_train)

                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(self.X_test)

                # Stores comprehensive results
                results[name] = {
                    'model': best_model,
                    'best_params': grid_search.best_params_,
                    'cv_scores': cv_scores,
                    'predictions': y_pred,
                    'report': classification_report(self.y_test, y_pred)
                }

                # Generate visualizations
                # Confusion matrix shows true vs predicted class distribution
                # Feature importance shows which input features had the most impact on predictions
                self._plot_confusion_matrix(self.y_test, y_pred, name)
                if hasattr(best_model, 'feature_importances_'):
                    self._plot_feature_importance(best_model, name)

                print(f"\n{name} Results:")
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
                print("\nClassification Report:")
                print(classification_report(self.y_test, y_pred))

            return results

        except Exception as e:
            print(f"Error in model training: {str(e)}")
            raise

    def _plot_confusion_matrix(self, y_true, y_pred, model_name):
        """
        Visual representation of prediction accuracy
                  True Positives (correctly predicted wins)
                  True Negatives (correctly predicted losses/draws)
                  False Positives (incorrectly predicted wins)
                  False Negatives (incorrectly predicted losses/draws)
                  Args:
                  y_true: the actual labels
                  y_pred: the predicted labels from the model
                  model_name: the name of the model for which the confusion matrix is being plotted
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Lost/Draw', 'Won'],
                yticklabels=['Lost/Draw', 'Won']
            )
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Outcome')
            plt.xlabel('Predicted Outcome')

            plt.savefig(
                os.path.join(self.output_dir, 'matrices', f'{model_name.lower()}_confusion_matrix.png'),
                bbox_inches='tight',
                dpi=300
            )
            plt.close()

        except Exception as e:
            print(f"Error creating confusion matrix plot: {str(e)}")
            raise

    def _plot_feature_importance(self, model, model_name):
        """
           Visualizes which features have the most impact on the model's predictions
           Feature importance rankings
        """
        try:
            # dataframe with feature names and their importance scores
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(12, 6))
            sns.barplot(data=importance.head(10), x='importance', y='feature')
            plt.title(f'Top 10 Most Important Features - {model_name}')
            plt.xlabel('Feature Importance')
            plt.tight_layout()

            plt.savefig(
                os.path.join(self.output_dir, 'feature_importance', f'{model_name.lower()}_feature_importance.png'),
                bbox_inches='tight',
                dpi=300
            )
            plt.close()

            print(f"\nTop 5 most important features for {model_name}:")
            print(importance.head().to_string(index=False))

        except Exception as e:
            print(f"Error in feature importance plotting: {str(e)}")
            raise


def main():
    """Main function"""
    try:
        # initializing predictor
        predictor = MatchPredictor()

        print("Loading and processing data...")
        predictor.load_and_process_data()

        print("\nPreparing features...")
        predictor.prepare_features()

        print("\nTraining models...")
        results = predictor.train_models()

        print("\nAnalysis completed successfully!")
        print(f"Results saved in: {predictor.output_dir}")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
