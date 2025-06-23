# CERT r4.2 Insider Threat Detection

This project explores the use of supervised learning and ensemble modeling to detect insider threats in organizational environments using the **CERT r4.2 dataset**. It leverages advanced feature engineering, multiple data balancing strategies, and a meta-learning ensemble to produce robust threat detection models.

---

## 📁 Project Structure

```
.
├── code_final.ipynb                 # Primary notebook: data processing, modeling & evaluation
├── feature_extraction.py           # Raw data parsing and weekly session feature engineering
├── meta_learner_tuning.py          # Model tuning using RandomizedSearchCV
├── Session1.ipynb                  # EDA notebook
├── r4-2-logon-eda.ipynb            # Logon event analysis
├── devicerr4_2.ipynb               # Device behavior analysis
├── 2Devicer4_2.ipynb               # Follow-up device behavior
├── 3devicer4_2.ipynb               # Additional device insights
```

---

## 🔍 Project Overview

Insider threats are among the hardest cybersecurity challenges to detect. These threats involve malicious behavior from individuals within an organization, such as employees or contractors, who exploit their access for harm. This project focuses on building a robust detection pipeline using machine learning, with the following key components:

- Rich session-level **feature engineering** from logon, file, device, email, and web activity data
- Training multiple **baseline classifiers**: CatBoost, XGBoost, Random Forest, and Decision Tree
- Construction of a **meta learner ensemble** for boosting predictive performance
- Use of **oversampling (SMOTE)** and **undersampling** to mitigate data imbalance
- Evaluation across multiple scenarios: user-agnostic models, feature selection, hybrid data transformations

---

## 🧪 Dataset

- **Source**: [CMU CERT Insider Threat Dataset R4.2](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099)
- Contains user activity logs from over 1000 employees, simulating real organizational behavior over 18 months
- Includes both normal and malicious user sessions

---

## ⚙️ Requirements

Install Python dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost imbalanced-learn tqdm joblib
```

---

## 🚀 How to Run

1. **Feature Extraction**
   ```bash
   python feature_extraction.py
   ```

2. **Model Training and Tuning**
   - Use `meta_learner_tuning.py` to tune baseline models and build meta learner
   - Track results in `code_final.ipynb`

3. **Exploratory Analysis**
   - EDA notebooks provide in-depth look into data patterns, anomalies, and event distributions

---

## 📈 Performance Metrics

From our experiments, we evaluated model performance across several scenarios. Here are selected results:

| Model         | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| CatBoost      | 97.14%    | 92.73% | 94.88%   |
| XGBoost       | 95.45%    | 91.00% | 93.46%   |
| Random Forest | 88.24%    | 80.91% | 83.94%   |
| Decision Tree | 86.11%    | 76.36% | 81.06%   |
| **Meta Learner** | **98.08%** | **92.73%** | **95.33%** |

The **meta learner consistently outperformed** individual classifiers across all scenarios.

---

## ✨ Key Highlights

- Developed a modular data pipeline to handle high-dimensional, temporal insider threat data
- Used feature selection (ANOVA + MI) to reduce noise and improve efficiency
- Created a user-agnostic model to generalize across unseen users
- Achieved a **highest F1-score of 96.26%** in Scenario 5 by combining feature selection with user-agnostic modeling

---

## 👤 Contributor

- **Raghav Seth**  
  Faculty of Engineering, University of Ottawa  
  rseth019@uottawa.ca

---

## 📄 License

This project is for educational purposes only. Use of the CERT dataset must follow the license terms defined by [CMU SEI](https://resources.sei.cmu.edu/).
