import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


class GoalAnalyzer:
    def __init__(self, data_dir=None):
        """
        Initialize the GoalAnalyzer with dynamic path handling.

        Args:
            data_dir (str, optional): Directory containing the data files.
                                    If None, uses default project structure.
        """
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # If no data_dir provided, construct it based on project structure
        if data_dir is None:
            # Navigate to the Data/InternationalMatches directory
            data_dir = os.path.join(script_dir, 'Data', 'InternationalMatches')

        # Set the file path for goalscorers.csv
        self.file_path = os.path.join(data_dir, 'goalscorers.csv')

        # Set up visualization directories relative to script location
        self.vis_dir = os.path.join(script_dir, 'Visualizations')
        self.matrix_dir = os.path.join(self.vis_dir, 'MatricesForInternationalGoalScorers')

        # Create directories if they don't exist
        os.makedirs(self.matrix_dir, exist_ok=True)

        # Initialize data containers
        self.df = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def load_and_clean_data(self):
        """
        Load and clean the dataset with comprehensive error handling and missing value treatment.
        """
        try:
            # Load data
            print(f"Loading data from: {self.file_path}")
            self.df = pd.read_csv(self.file_path)

            # Handle missing values
            print("\nChecking for missing values...")
            missing_stats = self.df.isnull().sum()
            print(missing_stats[missing_stats > 0])

            # Create numerical imputer for 'minute' column
            num_imputer = SimpleImputer(strategy='median')
            self.df['minute'] = num_imputer.fit_transform(self.df[['minute']])

            # Create categorical imputer for string columns
            cat_imputer = SimpleImputer(strategy='most_frequent')
            cat_columns = ['team', 'scorer', 'home_team', 'away_team']
            self.df[cat_columns] = cat_imputer.fit_transform(self.df[cat_columns])

            # Convert date and add features using NumPy
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['year'] = self.df['date'].dt.year
            self.df['month'] = self.df['date'].dt.month
            self.df['day_of_week'] = self.df['date'].dt.dayofweek

            # Add numpy-based features
            self.df['is_weekend'] = np.where(self.df['day_of_week'].isin([5, 6]), 1, 0)
            self.df['normalized_minute'] = (self.df['minute'] - np.mean(self.df['minute'])) / np.std(self.df['minute'])
            self.df['is_late_goal'] = np.where(self.df['minute'] >= 75, 1, 0)

            # Create binary features
            self.df['is_home_goal'] = np.where(self.df['team'] == self.df['home_team'], 1, 0)

            print("\nData cleaning completed successfully!")
            print(f"Final dataset shape: {self.df.shape}")

            return self.df

        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            print("Please ensure the data file exists in the correct location.")
            raise
        except Exception as e:
            print(f"Error during data loading and cleaning: {str(e)}")
            raise

    def prepare_ml_features(self):
        """
        Prepare features for machine learning with proper scaling and encoding.
        """
        try:
            # Create feature matrix using numpy operations
            numeric_features = ['minute', 'normalized_minute', 'month', 'day_of_week']
            categorical_features = ['team', 'scorer']

            # Encode categorical variables
            le_dict = {}
            encoded_features = []

            for cat_col in categorical_features:
                le_dict[cat_col] = LabelEncoder()
                encoded_col = le_dict[cat_col].fit_transform(self.df[cat_col])
                encoded_features.append(encoded_col)

            # Combine features using numpy
            X = np.column_stack([
                self.df[numeric_features].values,
                *encoded_features,
                self.df[['is_weekend', 'is_home_goal', 'penalty', 'own_goal']].values
            ])

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Create target variable (predicting if it's a late goal)
            y = self.df['is_late_goal'].values

            # Split data with stratification
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )

            print("\nFeature preparation completed successfully!")
            print(f"Training set shape: {self.X_train.shape}")
            print(f"Testing set shape: {self.X_test.shape}")

            return self.X_train, self.X_test, self.y_train, self.y_test

        except Exception as e:
            print(f"Error in feature preparation: {str(e)}")
            raise

    def train_models(self):
        """
        Train and evaluate two different ML models: Random Forest and Logistic Regression
        """
        try:
            # Initialize models
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )

            lr_model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )

            # Train both models
            print("\nTraining Random Forest Classifier...")
            rf_model.fit(self.X_train, self.y_train)
            rf_pred = rf_model.predict(self.X_test)

            print("Training Logistic Regression...")
            lr_model.fit(self.X_train, self.y_train)
            lr_pred = lr_model.predict(self.X_test)

            # Evaluate models
            print("\nRandom Forest Results:")
            print(classification_report(self.y_test, rf_pred))

            print("\nLogistic Regression Results:")
            print(classification_report(self.y_test, lr_pred))

            # Save confusion matrices
            self._plot_confusion_matrix(self.y_test, rf_pred, 'Random Forest')
            self._plot_confusion_matrix(self.y_test, lr_pred, 'Logistic Regression')

            return rf_model, lr_model

        except Exception as e:
            print(f"Error in model training: {str(e)}")
            raise

    def _plot_confusion_matrix(self, y_true, y_pred, model_name):
        """
        Helper method to plot and save confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model for the plot title
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            # Save in the matrix directory
            output_path = os.path.join(self.matrix_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
            plt.savefig(output_path)
            plt.close()

            print(f"Confusion matrix saved at: {output_path}")

        except Exception as e:
            print(f"Error creating confusion matrix plot: {str(e)}")
            raise


def main():
    """
    Main function to run the goal analysis pipeline.
    """
    try:
        # Initialize analyzer without specifying path - it will use default project structure
        analyzer = GoalAnalyzer()

        # Execute pipeline
        analyzer.load_and_clean_data()
        analyzer.prepare_ml_features()
        rf_model, lr_model = analyzer.train_models()

        print("\nAnalysis completed successfully!")
        print(f"Results saved in: {analyzer.matrix_dir}")

    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()