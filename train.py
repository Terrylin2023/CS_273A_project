import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve, average_precision_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import SelectFromModel
import warnings
import os
import time
from datetime import datetime
import json
import pickle
from scipy import stats
import torch
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


warnings.filterwarnings('ignore')
class TorchNN(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes=(100, 50)):
        super().__init__()
        layers = []
        prev_size = input_size
        
        # 建立隱藏層
        for h in hidden_sizes:
            layers.extend([
                torch.nn.Linear(prev_size, h),
                torch.nn.BatchNorm1d(h),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2)
            ])
            prev_size = h
        
        # 輸出層
        layers.append(torch.nn.Linear(prev_size, 2))
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

class NeuralNetworkClassifier:
    def __init__(self, input_size, hidden_sizes=(100, 50), batch_size=32, epochs=100, lr=0.001, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.model = TorchNN(input_size, hidden_sizes).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = None

    def get_params(self, deep=True):
        return {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'lr': self.lr,
            'device': self.device
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        # 重新建立模型
        self.model = TorchNN(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes
        ).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = None
        return self

    def fit(self, X, y):
        batch_size = self.batch_size
        epochs = self.epochs
        lr = self.lr
        # 將數據轉換為 numpy 數組
        X = np.array(X)
        y = np.array(y)
        # 轉換數據為 PyTorch tensors
        X = torch.FloatTensor(X).to(self.device)
        y = torch.LongTensor(y).to(self.device)
        
        # 創建數據加載器
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 優化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # 訓練循環
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        return self  # 返回 self
                
    def predict_proba(self, X):
        self.model.eval()
        X = np.array(X)
        X = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X)
            return torch.softmax(outputs, dim=1).cpu().numpy()
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        try:
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.api.types.is_categorical_dtype(obj):
                return str(obj)
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif hasattr(obj, '_asdict'):  # For namedtuples
                return obj._asdict()
            elif isinstance(obj, (pd.Int64Dtype, pd.Float64Dtype)):
                return str(obj)
            return super().default(obj)
        except:
            return str(obj)
class MLAnalysis:
    def __init__(self):
        """
        Initialize the ML Analysis class with detailed model configurations
        """
        self.create_directories()
        self.start_time = time.time()
        self.results_text = []
        self.feature_importance = None
        
        # Define hyperparameters with detailed explanations
        self.hyperparameters = {
            'random_forest': {
                'params': {
                    # 'n_estimators': [100, 200, 300],      # Number of trees
                    # 'max_depth': [10, 20, None],          # Maximum depth of trees
                    # 'min_samples_split': [2, 5, 10],      # Minimum samples required to split
                    # 'min_samples_leaf': [1, 2, 4],        # Minimum samples in leaf nodes
                    # 'max_features': ['sqrt', 'log2'],     # Feature selection method
                    # 'bootstrap': [True, False],           # Bootstrap samples
                    # 'class_weight': ['balanced', None]    # Class weight consideration
                    'n_estimators': [100,],      # Number of trees
                    'max_depth': [10,20],          # Maximum depth of trees
                    'class_weight': ['balanced']    # Class weight consideration
                },
                'param_explanations': {
                    'n_estimators': 'Controls the number of trees in the forest. More trees provide better accuracy but increase computation time.',
                    'max_depth': 'Maximum depth of each tree. None allows unlimited growth, while specific values prevent overfitting.',
                    'min_samples_split': 'Minimum samples required to split a node. Higher values prevent overfitting but might underfit.',
                    'min_samples_leaf': 'Minimum samples required in a leaf node. Higher values create more conservative trees.',
                    'max_features': 'Method for selecting features for splits. sqrt and log2 are common choices for classification.',
                    'bootstrap': 'Whether to use bootstrap samples. False means use whole dataset for each tree.',
                    'class_weight': 'Handling class imbalance. balanced adjusts weights inversely proportional to frequencies.'
                }
            },
            'xgboost': {
                'params': {
                    # 'learning_rate': [0.01, 0.1],         # Learning rate
                    # 'max_depth': [3, 5, 7],               # Maximum tree depth
                    # 'n_estimators': [100, 200],           # Number of boosting rounds
                    # 'subsample': [0.8, 1.0],              # Subsample ratio of training instances
                    # 'colsample_bytree': [0.8, 1.0],       # Subsample ratio of columns
                    # 'min_child_weight': [1, 3, 5],        # Minimum sum of instance weight in child
                    # 'gamma': [0, 0.1, 0.2],               # Minimum loss reduction for split
                    # 'reg_alpha': [0, 0.1, 0.5],           # L1 regularization
                    # 'reg_lambda': [0.1, 1.0]              # L2 regularization
                    'learning_rate': [0.1],         # Learning rate
                    'max_depth': [3, 6],               # Maximum tree depth
                    'n_estimators': [100],           # Number of boosting rounds
                  
                },
                'param_explanations': {
                    'learning_rate': 'Controls the contribution of each tree. Lower values mean more conservative boosting.',
                    'max_depth': 'Maximum depth of trees. Deeper trees can model more complex patterns but may overfit.',
                    'n_estimators': 'Number of boosting rounds. More rounds might improve performance but may overfit.',
                    'subsample': 'Fraction of samples used for training each tree. Helps prevent overfitting.',
                    'colsample_bytree': 'Fraction of features used for training each tree. Controls feature selection.',
                    'min_child_weight': 'Minimum sum of instance weight in child. Controls tree splitting behavior.',
                    'gamma': 'Minimum loss reduction required for split. Higher values make algorithm more conservative.',
                    'reg_alpha': 'L1 regularization term. Helps create sparse trees.',
                    'reg_lambda': 'L2 regularization term. Helps stabilize the model.'
                }
            },
            'lightgbm': {
                'params': {
                    # 'learning_rate': [0.01, 0.1],          # Learning rate
                    # 'num_leaves': [31, 63, 127],           # Maximum number of leaves
                    # 'max_depth': [3, 5, 7],                # Maximum tree depth
                    # 'n_estimators': [100, 200],            # Number of boosting iterations
                    # 'min_child_samples': [20, 50],         # Minimum samples in leaf
                    # 'min_child_weight': [0.001, 0.1],      # Minimum sum of instance weight
                    # 'min_split_gain': [0.0, 0.1],          # Minimum gain for split
                    # 'subsample': [0.8, 1.0],               # Sample ratio of training instances
                    # 'colsample_bytree': [0.8, 1.0],        # Feature selection ratio
                    # 'reg_alpha': [0.0, 0.1, 0.5],          # L1 regularization
                    # 'reg_lambda': [0.0, 0.1, 0.5],         # L2 regularization
                    # 'boosting_type': ['gbdt', 'dart']      # Boosting type
                    'learning_rate': [0.1],          # Learning rate
                    'max_depth': [3,6],                # Maximum tree depth
                    'n_estimators': [100],            # Number of boosting iterations
                    

                    'min_child_samples': [50],         # 增加最小樣本數要求
                    'min_child_weight': [0.01],        # 調整最小權重要求
                    'subsample': [0.8],                # 使用子採樣防止過擬合
                    'colsample_bytree': [0.8],         # 特徵採樣
                    'reg_alpha': [0.1],                # 增加一點 L1 正則化
                    'reg_lambda': [0.1],               # 增加一點 L2 正則化
                    'min_split_gain': [0.1]            # 設置最小分割增益
                },
                'param_explanations': {
                    'learning_rate': 'Step size shrinkage to prevent overfitting. Lower values need more iterations.',
                    'num_leaves': 'Maximum number of leaves in one tree. Controls model complexity.',
                    'max_depth': 'Maximum depth of the tree. -1 means no limit.',
                    'n_estimators': 'Number of boosting iterations. More iterations might improve performance.',
                    'min_child_samples': 'Minimum number of data needed in a leaf. Controls overfitting.',
                    'min_child_weight': 'Minimum sum of instance weight in leaf. Similar to min_child_samples.',
                    'min_split_gain': 'Minimum gain to make a split. Controls tree growth.',
                    'subsample': 'Training instance sampling ratio. Helps prevent overfitting.',
                    'colsample_bytree': 'Feature sampling ratio for each tree. Controls feature selection.',
                    'reg_alpha': 'L1 regularization. Helps create sparse trees.',
                    'reg_lambda': 'L2 regularization. Helps create more conservative trees.',
                    'boosting_type': 'Algorithm type. DART often provides better accuracy but might be unstable.'
                }
            },
            'neural_network': {
                'params': {
                    'hidden_sizes': [(50, 25), (128, 64, 32)],
                    'batch_size': [64],
                    'epochs': [100],
                    'lr': [ 0.0001]
                    
                    
                    # 'hidden_layer_sizes': [(50, 25)],  # Layer architecture
                    # 'activation': ['relu', 'tanh'],                        # Activation function
                    # 'alpha': [0.0001, 0.001, 0.01],                       # L2 penalty parameter
                    # 'learning_rate': ['constant', 'adaptive'],             # Learning rate schedule
                    # 'max_iter': [100],                               # Maximum iterations
                    # 'early_stopping': [True],                             # Early stopping usage
                    # 'validation_fraction': [0.1],                         # Validation set size
                    # 'batch_size': [ 64,128,256]                        # Batch size for training
                    # 'hidden_layer_sizes': [ (50, 25)],  # Layer architecture
                    # 'activation': ['relu'],                        # Activation function
                    # 'alpha': [0.0001],                       # L2 penalty parameter
                    # 'learning_rate': [ 'adaptive'],             # Learning rate schedule
                    # 'max_iter': [200],                               # Maximum iterations
                    # 'early_stopping': [True],                             # Early stopping usage
                    # 'validation_fraction': [0.1],                         # Validation set size
                    # 'batch_size': ['auto']                        # Batch size for training
                },
                'param_explanations': {
                    'hidden_layer_sizes': 'Architecture of hidden layers. More complex architectures can model more complex patterns.',
                    'activation': 'Activation function for hidden layers. ReLU is often default, tanh can work better for some cases.',
                    'alpha': 'L2 regularization term. Higher values mean stronger regularization.',
                    'learning_rate': 'Learning rate schedule for weight updates. Adaptive can be better for complex problems.',
                    'max_iter': 'Maximum number of iterations. Should be increased if model doesnt converge.',
                    'early_stopping': 'Whether to use early stopping to prevent overfitting.',
                    'validation_fraction': 'Fraction of training data to use for validation.',
                    'batch_size': 'Size of minibatches for training. Auto lets algorithm decide best size.'
                }
            }
        }
        
        # Initialize experiment tracking
        self.experiment_results = {
            'data_analysis': {},
            'feature_engineering': {},
            'model_training': {},
            'model_evaluation': {},
            'statistical_tests': {},
            'parameter_analysis': {}  # New section for parameter analysis
        }

    def create_directories(self):
        """
        Create necessary directories for storing results and artifacts:
        - plots: for all visualizations
        - results: for numerical results and reports
        - models: for saved model states
        - stats: for statistical analysis results
        """
        directories = ['plots', 'results', 'models', 'stats']
        for d in directories:
            if not os.path.exists(d):
                os.makedirs(d)

    def log(self, message, category=None):
        """
        Log messages and store them in appropriate categories.
        Args:
            message (str): The message to log
            category (str): The category of the log (e.g., 'data_analysis', 'model_training')
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_message = f"[{timestamp}] {message}"
        print(formatted_message)
        self.results_text.append(formatted_message)
        
        if category and category in self.experiment_results:
            if 'logs' not in self.experiment_results[category]:
                self.experiment_results[category]['logs'] = []
            self.experiment_results[category]['logs'].append(formatted_message)


    def load_data(self):
        """
        Load and perform initial analysis of the dataset:
        - Load training and test data
        - Analyze data quality
        - Perform basic statistical analysis
        - Save initial data analysis results
        """
        # Define column names
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
        
        # Load data
        self.train_data = pd.read_csv('dataset/adult.data', names=columns, skipinitialspace=True)
        self.test_data = pd.read_csv('dataset/adult.test', names=columns, skipinitialspace=True, skiprows=1)
        self.test_data['income'] = self.test_data['income'].str.replace('.', '')

        # Perform initial statistical analysis
        self.analyze_data_quality()
        self.perform_statistical_analysis()
        
        # Log results
        self.log(f"Training set shape: {self.train_data.shape}", 'data_analysis')
        self.log(f"Test set shape: {self.test_data.shape}", 'data_analysis')

    def analyze_data_quality(self):
        """
        Comprehensive data quality analysis:
        - Missing values analysis
        - Data type analysis
        - Uniqueness analysis
        - Basic statistics
        - Save results for reporting
        """
        quality_analysis = {
            'missing_values': {
                'train': self.train_data.isin(['?']).sum().to_dict(),
                'test': self.test_data.isin(['?']).sum().to_dict()
            },
            'data_types': {
                'train': self.train_data.dtypes.to_dict(),
                'test': self.test_data.dtypes.to_dict()
            },
            'unique_values': {
                'train': {col: self.train_data[col].nunique() 
                        for col in self.train_data.columns},
                'test': {col: self.test_data[col].nunique() 
                        for col in self.test_data.columns}
            }
        }
        
        # Save analysis results
        self.experiment_results['data_analysis']['quality_analysis'] = quality_analysis
        
        # Log key findings
        self.log("\nMissing Values Analysis:", 'data_analysis')
        self.log("\nTraining set missing values:", 'data_analysis')
        self.log(pd.Series(quality_analysis['missing_values']['train']).to_string(), 'data_analysis')
        self.log("\nTest set missing values:", 'data_analysis')
        self.log(pd.Series(quality_analysis['missing_values']['test']).to_string(), 'data_analysis')
        

    def save_experiment_state(self):
        """
        Save the current state of the experiment including:
        - All numerical results
        - Logs
        - Configuration
        - Timestamps
        """
        # Convert all numpy/pandas types to Python native types
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif pd.api.types.is_categorical_dtype(obj):
                return str(obj)
            elif isinstance(obj, (pd.Int64Dtype, pd.Float64Dtype)):
                return str(obj)
            return obj

        experiment_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'duration': time.time() - self.start_time,
            'results': convert_to_serializable(self.experiment_results),
            'configuration': {
                'hyperparameters': convert_to_serializable(self.hyperparameters)
            }
        }
        
        try:
            # Save as JSON using custom encoder
            with open('results/experiment_state.json', 'w') as f:
                json.dump(experiment_data, f, indent=4, cls=NumpyEncoder)
        except Exception as e:
            print(f"Warning: Could not save full experiment state due to: {str(e)}")
            # Fallback: Save basic information
            basic_data = {
                'timestamp': experiment_data['timestamp'],
                'duration': experiment_data['duration'],
                'configuration': {
                    'hyperparameters': self.hyperparameters
                }
            }
            with open('results/experiment_state_basic.json', 'w') as f:
                json.dump(basic_data, f, indent=4)
            print("Saved basic experiment state instead.")

    def create_statistical_plots(self):
        """
        Create and save statistical visualization plots:
        - Distribution plots
        - Correlation heatmap
        - Box plots
        - Violin plots
        """
        # 1. Distribution Plots
        numerical_features = self.train_data.select_dtypes(include=['int64', 'float64']).columns
        plt.figure(figsize=(20, 15))
        for idx, col in enumerate(numerical_features, 1):
            plt.subplot(3, 4, idx)
            sns.histplot(data=self.train_data, x=col, hue='income', multiple="stack")
            plt.title(f'{col} Distribution by Income')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/numerical_distributions.png')
        plt.close()
        
        # 2. Correlation Heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.train_data[numerical_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.savefig('plots/correlation_heatmap.png')
        plt.close()
        
        # 3. Box Plots for Numerical Features
        plt.figure(figsize=(20, 15))
        for idx, col in enumerate(numerical_features, 1):
            plt.subplot(3, 4, idx)
            sns.boxplot(data=self.train_data, x='income', y=col)
            plt.title(f'{col} by Income')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/boxplots.png')
        plt.close()

    def log_statistical_findings(self, stats_results):
        """
        Log key statistical findings and insights
        """
        # Log normality test results
        self.log("\nNormality Test Results:", 'statistical_tests')
        for feature, result in stats_results['normality_tests'].items():
            self.log(f"{feature}: p-value = {result['p_value']:.4f}", 'statistical_tests')
        
        # Log independence test results
        self.log("\nFeature Independence Test Results:", 'statistical_tests')
        for feature, result in stats_results['independence_tests'].items():
            self.log(f"{feature}: chi2 = {result['chi2']:.4f}, p-value = {result['p_value']:.4f}", 
                    'statistical_tests')
        
        # Log correlation findings
        self.log("\nStrong Correlations (|r| > 0.5):", 'statistical_tests')
        correlation_matrix = pd.DataFrame(stats_results['correlation'])
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i,j]) > 0.5:
                    strong_correlations.append(
                        f"{correlation_matrix.columns[i]} - {correlation_matrix.columns[j]}: "
                        f"{correlation_matrix.iloc[i,j]:.4f}"
                    )
        for corr in strong_correlations:
            self.log(corr, 'statistical_tests')

    def feature_engineering(self):
        """
        Comprehensive feature engineering process:
        1. Handle missing values
        2. Create new features
        3. Transform existing features
        4. Encode categorical variables
        5. Analyze feature importance
        
        All steps are documented and results are saved
        """
        feature_engineering_results = {}
        
        # 1. Handle missing values
        for data in [self.train_data, self.test_data]:
            data.replace('?', np.nan, inplace=True)
            for col in ['workclass', 'occupation', 'native-country']:
                data[col].fillna(data[col].mode()[0], inplace=True)
        
        # 2. Create new features
        for data in [self.train_data, self.test_data]:
            # Financial features
            data['capital_total'] = data['capital-gain'] - data['capital-loss']
            data['has_capital'] = (data['capital_total'] != 0).astype(int)
            data['capital_per_hour'] = data['capital_total'] / (data['hours-per-week'] + 1)
            
            # Work-related features
            data['work_intensity'] = data['hours-per-week'] / data['age']
            
            # Education mapping
            education_map = {
                'Preschool': 1, '1st-4th': 1, '5th-6th': 1, '7th-8th': 2, '9th': 2,
                '10th': 2, '11th': 2, '12th': 2, 'HS-grad': 3, 'Some-college': 3,
                'Assoc-voc': 4, 'Assoc-acdm': 4, 'Bachelors': 5, 'Masters': 6,
                'Prof-school': 7, 'Doctorate': 7
            }
            data['education_level'] = data['education'].map(education_map)
        
        # Record feature engineering steps
        feature_engineering_results['new_features'] = {
            'financial_features': ['capital_total', 'has_capital', 'capital_per_hour'],
            'work_features': ['work_intensity'],
            'education_features': ['education_level']
        }
        
        # Analyze new features
        self.analyze_engineered_features()
        
        # Save results
        self.experiment_results['feature_engineering'] = feature_engineering_results
        
        # Log feature engineering summary
        self.log("\nFeature Engineering:", 'feature_engineering')
        self.log("- Created financial features: capital_total, has_capital, capital_per_hour", 
                'feature_engineering')
        self.log("- Created work-related features: work_intensity", 'feature_engineering')
        self.log("- Created education level mapping", 'feature_engineering')
    def analyze_engineered_features(self):
        """
        Analyze the effectiveness of engineered features:
        - Statistical analysis of new features
        - Correlation with target variable
        - Feature importance analysis
        - Visualization of feature distributions
        """
        new_features = ['capital_total', 'has_capital', 'capital_per_hour', 
                    'work_intensity', 'education_level']
        
        analysis_results = {}
        
        # Statistical analysis of new features
        for feature in new_features:
            analysis_results[feature] = {
                'statistics': self.train_data[feature].describe().to_dict(),
                'correlation_with_target': stats.pointbiserialr(
                    self.train_data[feature],
                    (self.train_data['income'] == '>50K').astype(int)
                )._asdict()
            }
        
        # Visualize new features
        plt.figure(figsize=(15, 10))
        for idx, feature in enumerate(new_features, 1):
            plt.subplot(2, 3, idx)
            sns.boxplot(data=self.train_data, x='income', y=feature)
            plt.title(f'{feature} by Income')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/engineered_features_analysis.png')
        plt.close()
        
        # Save analysis results
        self.experiment_results['feature_engineering']['feature_analysis'] = analysis_results

    def prepare_data(self):
        """
        Prepare data for model training:
        - Encode categorical variables
        - Scale numerical features
        - Handle class imbalance
        - Split features and target
        """
        # Encode categorical variables
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'native-country']
        
        self.encoders = {}
        for col in categorical_cols:
            self.encoders[col] = LabelEncoder()
            self.train_data[col] = self.encoders[col].fit_transform(self.train_data[col])
            self.test_data[col] = self.encoders[col].transform(self.test_data[col])
        
        # Select features
        self.feature_cols = ['age', 'workclass', 'education-num', 'education_level',
                        'marital-status', 'occupation', 'relationship', 'race', 'sex',
                        'capital_total', 'has_capital', 'work_intensity', 
                        'capital_per_hour', 'hours-per-week']
        
        # Prepare X and y
        X_train = self.train_data[self.feature_cols]
        y_train = (self.train_data['income'] == '>50K').astype(int)
        X_test = self.test_data[self.feature_cols]
        y_test = (self.test_data['income'] == '>50K').astype(int)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Handle imbalanced data
        smote_tomek = SMOTETomek(random_state=42)
        self.X_train_balanced, self.y_train_balanced = smote_tomek.fit_resample(X_train_scaled, y_train)
        self.X_test, self.y_test = X_test_scaled, y_test
        
        # Save preparation results
        self.experiment_results['data_preparation'] = {
            'feature_columns': self.feature_cols,
            'training_shape': self.X_train_balanced.shape,
            'test_shape': self.X_test.shape,
            'class_distribution': {
                'original': np.bincount(y_train).tolist(),
                'balanced': np.bincount(self.y_train_balanced).tolist()
            }
        }
        
        self.log("\nData Preparation:", 'data_preparation')
        self.log(f"Number of features: {len(self.feature_cols)}", 'data_preparation')
        self.log(f"Training set balanced shape: {self.X_train_balanced.shape}", 'data_preparation')
        self.log(f"Test set shape: {self.X_test.shape}", 'data_preparation')
    
    def create_model_comparison_plots(self, model_results):
        """
        Create visualization plots comparing model performance
        """
        # Model accuracy comparison
        accuracies = {
            name: results['test_performance']['accuracy'] 
            for name, results in model_results.items()
        }
        
        plt.figure(figsize=(10, 6))
        plt.bar(accuracies.keys(), accuracies.values())
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/model_comparison.png')
        plt.close()

    def create_model_performance_plots(self, model_name, results, model=None):
        """
        Create detailed performance plots for a single model
        Args:
            model_name: Name of the model
            results: Dictionary containing model results
            model: Optional model object (for making predictions)
        """
        try:
            # Create feature importance plot if available
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                plt.figure(figsize=(10, 6))
                sns.barplot(data=importance_df.head(10), x='importance', y='feature')
                plt.title(f'{model_name} - Feature Importance')
                plt.tight_layout()
                plt.savefig(f'plots/{model_name.lower()}_feature_importance.png')
                plt.close()
            
            # Create confusion matrix plot
            y_true = self.y_test
            y_pred = np.array([1 if p > 0.5 else 0 for p in results['test_performance']['classification_report']['1']['precision']])
            
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{model_name} - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'plots/{model_name.lower()}_confusion_matrix.png')
            plt.close()
            
        except Exception as e:
            self.log(f"Error creating performance plots for {model_name}: {str(e)}", 'model_training')

    def train_and_evaluate_model(self, model, name, params=None):
        """
        Train and evaluate a single model with improved settings
        """
        results = {}
        
        try:
            if params:
                grid_search = GridSearchCV(
                    model, 
                    params['params'],
                    cv=3,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=1
                )
                
                # Fit model
                grid_search.fit(self.X_train_balanced, self.y_train_balanced)
                
                # Store grid search results
                results['grid_search_results'] = {
                    'params': grid_search.cv_results_['params'],
                    'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                    'std_test_score': grid_search.cv_results_['std_test_score'].tolist()
                }
                
                model = grid_search.best_estimator_
                results['best_parameters'] = grid_search.best_params_
            else:
                model.fit(self.X_train_balanced, self.y_train_balanced)
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, 
                self.X_train_balanced, 
                self.y_train_balanced, 
                cv=5
            )
            
            results['cross_validation'] = {
                'scores': cv_scores.tolist(),
                'mean': cv_scores.mean(),
                'std': cv_scores.std()
            }
            
            # Evaluate on test set
            y_pred = model.predict(self.X_test)
            
            # y_pred = best_model.predict(self.X_test)
            test_accuracy = accuracy_score(self.y_test, y_pred)
            results['test_performance'] = {
                'accuracy': test_accuracy,
                'classification_report': classification_report(
                    self.y_test, 
                    y_pred, 
                    output_dict=True
                )
            }
            
            # Create performance plots with the model object
            self.create_model_performance_plots(name, results, model)
            
            return results
            
        except Exception as e:
            self.log(f"Error in model training and evaluation: {str(e)}", 'model_training')
            return None
    def train_and_evaluate_models(self):
        """
        Train and evaluate multiple models with neural network support
        """
        # 定義基本模型
        models = {
            'random_forest': (RandomForestClassifier(random_state=42), self.hyperparameters['random_forest']),
            'xgboost': (xgb.XGBClassifier(random_state=42), self.hyperparameters['xgboost']),
            'lightgbm': (lgb.LGBMClassifier(random_state=42), self.hyperparameters['lightgbm'])
        }
        
        self.log("\nModel Training and Evaluation:", 'model_training')
        model_results = {}
        
        # 訓練基本模型
        for name, (model, params) in models.items():
            self.log(f"\nTraining {name}:", 'model_training')
            results = self.train_and_evaluate_model(model, name, params)
            model_results[name] = results
            
            # Log results
            self.log(f"Cross-validation scores: {results['cross_validation']['scores']}", 'model_training')
            self.log(f"Mean CV score: {results['cross_validation']['mean']:.4f} "
                    f"(+/- {results['cross_validation']['std'] * 2:.4f})", 'model_training')
            self.log(f"Test set accuracy: {results['test_performance']['accuracy']:.4f}", 'model_training')
            self.log("\nClassification Report:", 'model_training')
            self.log(pd.DataFrame(results['test_performance']['classification_report']).transpose().to_string(), 
                    'model_training')
        
        # 訓練神經網路
        if 'neural_network' in self.hyperparameters:
            try:
                self.log("\nTraining neural network:", 'model_training')
                nn_params = self.hyperparameters['neural_network']
                
                param_grid = {
                    'hidden_sizes': nn_params['params']['hidden_sizes'],
                    'batch_size': nn_params['params']['batch_size'],
                    'epochs': nn_params['params']['epochs'],
                    'lr': nn_params['params']['lr']
                }
                
                model = NeuralNetworkClassifier(
                    input_size=self.X_train_balanced.shape[1]
                )
                
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,
                    scoring='accuracy',
                    n_jobs=1,  # 避免序列化問題
                    verbose=1,
                    error_score='raise'
                )
                
                grid_search.fit(self.X_train_balanced, self.y_train_balanced)
                
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_
                
                y_pred = best_model.predict(self.X_test)
                test_accuracy = accuracy_score(self.y_test, y_pred)
                
                results = {
                    'grid_search_results': {
                        'params': grid_search.cv_results_['params'],
                        'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                        'std_test_score': grid_search.cv_results_['std_test_score'].tolist()
                    },
                    'best_parameters': best_params,
                    'cross_validation': {
                        'mean': best_score,
                        'std': grid_search.cv_results_['std_test_score'][grid_search.best_index_]
                    },
                    'test_performance': {
                        'accuracy': test_accuracy,
                        'classification_report': classification_report(
                            self.y_test,
                            y_pred,
                            output_dict=True
                        )
                    }
                }
                
                model_results['neural_network'] = results
                self.log(f"Neural network test accuracy: {test_accuracy:.4f}", 'model_training')
                self.log(f"Neural network best parameters: {best_params}", 'model_training')
            except Exception as e:
                self.log(f"Error training neural network: {str(e)}", 'model_training')

        # 保存結果並創建可視化
        self.experiment_results['model_training']['model_results'] = model_results
        self.create_model_comparison_plots(model_results)

    def create_model_comparison_plots(self, model_results):
        """
        Create visualization plots comparing model performance
        """
        # Model accuracy comparison
        accuracies = {
            name: results['test_performance']['accuracy'] 
            for name, results in model_results.items()
        }
        
        plt.figure(figsize=(10, 6))
        plt.bar(accuracies.keys(), accuracies.values())
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/model_comparison.png')
        plt.close()

    def create_model_performance_plots(self, model_name, results, model=None):
        """
        Create detailed performance plots for a single model
        Args:
            model_name: Name of the model
            results: Dictionary containing model results
            model: Optional model object (for making predictions)
        """
        try:
            # Create feature importance plot if available
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                plt.figure(figsize=(10, 6))
                sns.barplot(data=importance_df.head(10), x='importance', y='feature')
                plt.title(f'{model_name} - Feature Importance')
                plt.tight_layout()
                plt.savefig(f'plots/{model_name.lower()}_feature_importance.png')
                plt.close()
            
            # Create confusion matrix plot
            y_true = self.y_test
            y_pred = np.array([1 if p > 0.5 else 0 for p in results['test_performance']['classification_report']['1']['precision']])
            
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{model_name} - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'plots/{model_name.lower()}_confusion_matrix.png')
            plt.close()
            
        except Exception as e:
            self.log(f"Error creating performance plots for {model_name}: {str(e)}", 'model_training')

    def perform_statistical_analysis(self):
            """
            Comprehensive statistical analysis of the dataset:
            - Descriptive statistics
            - Distribution analysis
            - Correlation analysis
            - Hypothesis testing
            - Feature relationships
            """
            stats_results = {}
            
            # 1. Descriptive Statistics
            numerical_features = self.train_data.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = self.train_data.select_dtypes(include=['object']).columns
            
            stats_results['descriptive'] = {
                'numerical': {
                    'train': self.train_data[numerical_features].describe().to_dict(),
                    'test': self.test_data[numerical_features].describe().to_dict()
                },
                'categorical': {
                    'train': {col: self.train_data[col].value_counts().to_dict() 
                            for col in categorical_features},
                    'test': {col: self.test_data[col].value_counts().to_dict() 
                            for col in categorical_features}
                }
            }
            
            # 2. Normality Tests
            stats_results['normality_tests'] = {}
            for col in numerical_features:
                stat, p_value = stats.normaltest(self.train_data[col].dropna())
                stats_results['normality_tests'][col] = {
                    'statistic': stat,
                    'p_value': p_value
                }
            
            # 3. Feature Independence Tests (Chi-square)
            stats_results['independence_tests'] = {}
            for col in categorical_features:
                if col != 'income':
                    contingency_table = pd.crosstab(self.train_data[col], self.train_data['income'])
                    chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
                    stats_results['independence_tests'][col] = {
                        'chi2': chi2,
                        'p_value': p_value
                    }
            
            # 4. Correlation Analysis
            correlation_matrix = self.train_data[numerical_features].corr()
            stats_results['correlation'] = correlation_matrix.to_dict()
            
            # Save results
            self.experiment_results['statistical_tests'] = stats_results
            
            # Create visualizations
            self.create_statistical_plots()
            
            # Log key findings
            self.log_statistical_findings(stats_results)

    def analyze_parameter_impact(self, model_name, model_results):
        """
        Analyze the impact of different parameters on model performance
        """
        param_analysis = {
            'parameter_importance': {},
            'best_params_analysis': {},
            'param_performance_correlation': {}
        }
        
        # Get parameter trial results from grid search
        if 'best_parameters' in model_results:
            grid_results = model_results['grid_search_results']
            params_tested = self.hyperparameters[model_name]['params']
            
            # Analyze each parameter's impact
            for param_name in params_tested.keys():
                param_values = []
                scores = []
                for params, score in zip(grid_results['params'], grid_results['mean_test_score']):
                    if param_name in params:
                        param_values.append(params[param_name])
                        scores.append(score)
                
                # Calculate parameter importance
                if len(set(param_values)) > 1:
                    correlation = stats.spearmanr(param_values, scores)[0]
                    param_analysis['parameter_importance'][param_name] = abs(correlation)
                
            # Analyze best parameters
            best_params = model_results['best_parameters']
            param_explanations = self.hyperparameters[model_name.lower()]['param_explanations']
            
            for param, value in best_params.items():
                param_analysis['best_params_analysis'][param] = {
                    'value': value,
                    'explanation': param_explanations[param],
                    'compared_to': f"Tested values: {params_tested[param]}"
                }
        
        return param_analysis

    def generate_model_report(self, model_name, model_results, param_analysis):
        """
        Generate comprehensive report for model performance and parameter selection
        """

        report = [
            f"\n{'='*80}",
            f"\nDETAILED REPORT FOR {model_name.upper()}",
            f"{'='*80}\n",
            
            "\n1. OVERALL PERFORMANCE",
            f"{'-'*50}",
        ]

        if 'cross_validation' in model_results:
            report.extend([
                f"Mean CV Score: {model_results['cross_validation']['mean']:.4f} (±{model_results['cross_validation']['std']*2:.4f})",
            ])
        
        report.extend([
            f"Test Accuracy: {model_results['test_performance']['accuracy']:.4f}",
            "\nClassification Report:",
            pd.DataFrame(model_results['test_performance']['classification_report']).to_string(),
            
            "\n2. PARAMETER ANALYSIS",
            f"{'-'*50}",
            "\nBest Parameters Selected:",
        ])
        
        # Add parameter analysis
        if 'best_params_analysis' in param_analysis:
            for param, analysis in param_analysis['best_params_analysis'].items():
                report.extend([
                    f"\n{param}:",
                    f"  Selected value: {analysis['value']}",
                    f"  Explanation: {analysis['explanation']}",
                    f"  Options considered: {analysis['compared_to']}"
                ])
        
        # Add parameter importance if available
        if 'parameter_importance' in param_analysis:
            report.extend([
                "\nParameter Importance Ranking:",
                "-" * 30
            ])
            
            # Sort parameters by importance
            sorted_params = sorted(
                param_analysis['parameter_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for param, importance in sorted_params:
                report.append(f"{param}: {importance:.4f}")
        
        # Join all parts of the report
        return "\n".join(report)

    def save_analysis_reports(self):
        """
        Save comprehensive analysis reports for all models
        """
        # Create main report file
        report_path = "results/model_analysis_report.txt"
        
        with open(report_path, 'w') as f:
            # Write overall summary
            f.write("MACHINE LEARNING MODEL ANALYSIS REPORT\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Write model-specific reports
            comparison_data = {}
            for model_name, results in self.experiment_results['model_training']['model_results'].items():
                model_data = {
                    'Test Accuracy': results['test_performance']['accuracy']
                }
                if 'cross_validation' in results:
                    model_data['CV Mean Score'] = results['cross_validation']['mean']
                    model_data['CV Std'] = results['cross_validation']['std']
                comparison_data[model_name] = model_data

            comparison_df = pd.DataFrame(comparison_data).T
            
            # Write comparison summary
            f.write("\nMODEL COMPARISON SUMMARY\n")
            f.write("="*80 + "\n")

            
            f.write(comparison_df.to_string())

def main():
    """Main execution function"""
    analysis = MLAnalysis()
    
    # Execute analysis pipeline
    analysis.load_data()
    analysis.feature_engineering()
    analysis.prepare_data()
    analysis.train_and_evaluate_models()
    analysis.save_experiment_state()
    analysis.save_analysis_reports()
    
    # Print execution summary
    execution_time = time.time() - analysis.start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()