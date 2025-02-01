import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from catboost import CatBoostClassifier

class TitanicPreprocessor:
    def __init__(self):
        self.categorical_features = ['Sex', 'Embarked', 'Pclass', 'Title', 'Deck']
        self.numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
        self.label_encoders = {}
    
    def extract_title(self, name):
        title = name.split(', ')[1].split('.')[0]
        rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 
                      'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        return 'Rare' if title in rare_titles else title
    
    def create_features(self, df):
        """Create all feature engineering in one place"""
        df = df.copy()
        
        # Extract title from name
        df['Title'] = df['Name'].apply(self.extract_title)
        
        # Cabin features
        df['Deck'] = df['Cabin'].fillna('M').str[0]
        df['HasCabin'] = df['Cabin'].notna().astype(int)
        
        # Family features
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # Fare features
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        df['Fare_Per_Person'] = df['Fare'] / df['FamilySize']
        
        # Age interaction
        df['Age*Class'] = df['Age'] * df['Pclass']
        
        return df
    
    def fit_transform(self, df):
        """Full preprocessing pipeline for training data"""
        # Create features
        df = self.create_features(df)
        
        # Handle missing values
        for col in self.numerical_features:
            df[col] = df[col].fillna(df[col].median())
            
        for col in self.categorical_features:
            df[col] = df[col].fillna(df[col].mode()[0])
            
        # Encode categorical features
        for col in self.categorical_features:
            self.label_encoders[col] = LabelEncoder()
            df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])
        
        return df
    
    def transform(self, df):
        """Transform test data using fitted preprocessor"""
        # Create features
        df = self.create_features(df)
        
        # Handle missing values similarly to training
        for col in self.numerical_features:
            df[col] = df[col].fillna(df[col].median())
            
        for col in self.categorical_features:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        # Use fitted label encoders
        for col in self.categorical_features:
            df[col + '_encoded'] = self.label_encoders[col].transform(df[col])
        
        return df

def train_models(X_train, y_train):
    """Train multiple models with cross-validation"""
    models = {
        'catboost': CatBoostClassifier(
            iterations=1000,
            learning_rate=0.02,
            depth=8,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
    }
    
    trained_models = {}
    for name, model in models.items():
        # Perform cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"{name} CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        # Train on full training data
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def main():
    # Load data
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Initialize preprocessor
    preprocessor = TitanicPreprocessor()
    
    # Preprocess training data
    processed_train = preprocessor.fit_transform(train_df)
    
    # Prepare features for training
    feature_cols = ([f"{col}_encoded" for col in preprocessor.categorical_features] + 
                   preprocessor.numerical_features + 
                   ['HasCabin', 'IsAlone', 'Fare_Per_Person', 'Age*Class'])
    
    X_train = processed_train[feature_cols]
    y_train = processed_train['Survived']
    
    # Train models
    trained_models = train_models(X_train, y_train)
    
    # Process test data
    processed_test = preprocessor.transform(test_df)
    X_test = processed_test[feature_cols]
    
    # Make predictions
    weights = {'catboost': 0.5, 'random_forest': 0.3, 'gradient_boosting': 0.2}
    weighted_predictions = np.zeros(len(X_test))
    
    for name, model in trained_models.items():
        weighted_predictions += model.predict_proba(X_test)[:, 1] * weights[name]
    
    final_predictions = (weighted_predictions > 0.5).astype(int)
    
    # Create submission file
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': final_predictions
    })
    submission.to_csv('submission.csv', index=False)
    
    # Print training metrics
    print("\nTraining Metrics:")
    for name, model in trained_models.items():
        train_preds = model.predict(X_train)
        print(f"\n{name} Performance:")
        print(classification_report(y_train, train_preds))

if __name__ == "__main__":
    main()