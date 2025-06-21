import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import RandomizedSearchCV

from sklearn.feature_selection import f_classif, mutual_info_classif
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def rank_and_plot_top_features(X, y, top_k=20, return_df=False):
    """
    Computes ANOVA F-score, Mutual Information, and Combined Rank for features,
    and plots top-k features by each method.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or array-like
        Target variable.
    top_k : int, default=20
        Number of top features to display in the plot.
    return_df : bool, default=False
        If True, returns top_k feature names and full ranked DataFrame.

    Returns
    -------
    top_features : list
        List of top_k features by Combined Rank.
    ranked_df : pd.DataFrame (optional)
        Full table with ANOVA, MI, and combined rankings.
    """
    # Step 1: Calculate scores
    f_scores, _ = f_classif(X, y)
    mi_scores = mutual_info_classif(X, y, random_state=42)

    ranked_df = pd.DataFrame({
        'Feature': X.columns,
        'ANOVA_F': f_scores,
        'MI': mi_scores
    })

    # calculate ranks
    ranked_df['ANOVA_Rank'] = ranked_df['ANOVA_F'].rank(ascending=False)
    ranked_df['MI_Rank'] = ranked_df['MI'].rank(ascending=False)
    ranked_df['Combined_Rank'] = (ranked_df['ANOVA_Rank'] + ranked_df['MI_Rank']) / 2
    ranked_df = ranked_df.sort_values(by='Combined_Rank').reset_index(drop=True)

    # select top-k
    top_combined = ranked_df.head(top_k)
    top_features = top_combined['Feature'].tolist()

    # prepare data for plots
    top_anova = ranked_df.sort_values('ANOVA_F', ascending=False).head(top_k)
    top_mi = ranked_df.sort_values('MI', ascending=False).head(top_k)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    sns.set(style="whitegrid")

    sns.barplot(ax=axs[0], x='ANOVA_F', y='Feature', data=top_anova, palette="Blues_r")
    axs[0].set_title(f"Top {top_k} Features by ANOVA F-score")

    sns.barplot(ax=axs[1], x='MI', y='Feature', data=top_mi, palette="Greens_r")
    axs[1].set_title(f"Top {top_k} Features by Mutual Information")

    sns.barplot(ax=axs[2], x='Combined_Rank', y='Feature', data=top_combined, palette="Purples_r")
    axs[2].invert_xaxis()  # Lower rank = more important
    axs[2].set_title(f"Top {top_k} Features by Combined Rank")

    plt.tight_layout()
    plt.show()

    if return_df:
        return top_features, ranked_df
    return top_features

def create_folds(df, smote, undersampling, test_size=0.1, num_folds=5, smote_ratio=0.1, undersampling_ratio=0.5):
    """
        Apply train-test split. Then apply k-fold on train data
        Each fold has 100% of total class-1 instance

        Params
        -------
        df: DataFrame with feature and target columns
            unnecessary columns like week, user must be removed already
        smote: bool
            if True, apply smote after train_test split
        test_size: test size for train-test split
        num_folds: Number of folds for K-fold

        Returns
        -------
        final_folds: list
            List of all folds. Each element of this list is another list containing 4 dataframes
            format: [
            [fold_1_Xtrain, fold_1_ytrain, fold_1_Xval, fold_1_yval], 
            [fold_2_Xtrain, fold_2_ytrain, fold_2_Xval, fold_2_yval], 
            []......
                    ]
    """
    final_folds = []

    features = df.drop(columns=['insider'])
    target = df['insider']
    target.loc[~target.isin([0,1])] = 1

    # Step 1: Stratified Train-Test Split (10% test set)
    X, X_test, y, y_test = train_test_split(
        features, target, test_size=test_size, random_state=42, stratify=target
    )
    X.reset_index(inplace=True, drop=True)
    y=y.reset_index()
    y=y['insider']

    # Step 2: K-Fold Cross-Validation on Train Set (Keeping All Class 1 + SMOTE)
    num_folds = num_folds
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # # Identify all class 1 samples in training set
    # class_1_indices = np.where(y != 0)[0]
    
    for train_index, val_index in skf.split(X, y):
        # # Ensure all class 1 samples are included in the training set
        # train_index = np.concatenate([train_index, class_1_indices])
        # train_index = np.unique(train_index)  # Remove duplicates
    
        X_train_fold, X_val_fold = X.loc[train_index], X.loc[val_index]
        y_train_fold, y_val_fold = y.loc[train_index], y.loc[val_index]
    
        if smote:
            # Apply SMOTE on the training fold
            smote = SMOTE(sampling_strategy=smote_ratio, random_state=42)  # Adjust as needed
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, y_train_fold)
        else:
            X_train_resampled = X_train_fold
            y_train_resampled = y_train_fold
        
        if undersampling:
            # undersampling
            undersampler = RandomUnderSampler(sampling_strategy=undersampling_ratio, random_state=42)  # Adjust ratio
            X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train_resampled, y_train_resampled)
    
        final_folds.append([X_train_resampled, y_train_resampled, X_val_fold, y_val_fold])

    return final_folds, [X_test, y_test]

def tune_model(param_grid, model_name, X=X_top50, y=y_top50):
    """
        Tune given model
    """
    if model_name=='xgb':
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    elif model_name=='cat':
        model = CatBoostClassifier()
    elif model_name=='rf':
        model = RandomForestClassifier()
    else:
        model = DecisionTreeClassifier()

    random_search = RandomizedSearchCV(
        model, param_distributions=param_grid, n_iter=30, scoring='recall', cv=5, verbose=2, n_jobs=-1, random_state=42
        )

    random_search.fit(X, y)
    print(f"{model_name} Best Parameters:", random_search.best_params_)

    with open(f'{model_name}_tuning_result.pickle','wb') as file:
        pickle.dump(random_search, file)

    print(f"{model_name} Tuning completed...")

DATA_DIR = os.path.join(os.getcwd(),'feature-extraction-for-CERT-insider-threat-test-datasets-main','ExtractedData','merged')

df_base = pd.read_csv(os.path.join(DATA_DIR,'session.csv'))
# df_base = pd.read_csv(session_path)
df_base = df_base.drop(columns=['Unnamed: 0','starttime','endtime','sessionid','week'])

X = df_base.drop(columns=['insider'])
y = df_base['insider']
top_features = rank_and_plot_top_features(X, y, top_k=50)

df_fs_2 = pd.read_csv(os.path.join(DATA_DIR,'session.csv'))
df_fs_2 = df_fs_2.drop(columns=['Unnamed: 0','starttime','endtime','sessionid','week'])

X = df_fs_2.drop(columns=['insider'])
y = df_fs_2['insider']
y.loc[~y.isin([0,1])] = 1

X_top50 = X[top_features].drop(columns=['user'], errors='ignore')
y_top50 = y

model_names = ['xgb','cat','rf','dt']
param_grids = {
    'xgb': {
        'n_estimators': [100, 300, 500, 800],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.001, 0.01, 0.05, 0.1],
        'min_child_weight': [1, 3, 5, 7],
        'subsample': [0.5, 0.7, 0.8, 1.0],
        'colsample_bytree': [0.5, 0.7, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.1, 0.5, 1],  # L1 Regularization
        'reg_lambda': [0, 0.1, 0.5, 1],  # L2 Regularization
    },
    'cat': {
        'iterations': [100, 200, 500, 1000],
        'learning_rate': [0.01, 0.03, 0.1, 0.3],
        'depth': [4, 6, 8, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'border_count': [32, 64, 128],
        'bagging_temperature': [0.0, 0.5, 1.0]
    },
    'rf': {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False]
    },
    'dt': {
        'max_depth': [None, 5, 10, 15, 20, 25, 30],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy']
    }
    }

for model_name in model_names:
    tune_model(
        param_grid=param_grids[model_name],
        model_name=model_name
)