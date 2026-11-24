# **Football Players Market Value Predictor using a Multilayer Perceptron**

---

Predicting the market value of football players based on their stats over three seasons, using a Multilayer Perceptron (MLP) built with PyTorch and trained on a dataset I made myself.

**Author:** Ayman Bohsali

**Course Professor:** Dr. Mirco Ravanelli

**Project Advisor:** Gianfranco Dumoulin

---

## **Overview**

Football clubs rely more and more on data when it comes to recruitment, scouting, and contract negotiations. The problem is that while many websites publish market values for players, there is little consistency between them. These values are often influenced by personal biases, crowdsourced opinions, or undisclosed models.

The goal of this project is to predict a player's market value from his stats, age, nationality, position, team, league, and contract duration. I chose a Multilayer Perceptron because the relationship between these features and the market value is complex and non-linear. For example, a young player with a certain number of goals could be worth a lot more than an older player with the same number of goals.

---

## **The Dataset**

I couldn't find a ready-made dataset that had exactly what I wanted, so I made my own by combining data from:

- **[FBref](https://fbref.com/)**: player statistics for multiple seasons
- **[Transfermarkt](https://www.transfermarkt.com/)**: market values and contract expiration dates

The datasets I found on Kaggle were either for a single season, had different columns from one season to another, or didn't include contract and market value information. Other models I found online either used FIFA video game ratings as input (which don't really reflect how well a player is currently doing) or only used data from one season.

What makes this dataset different is that it covers **three consecutive seasons (2022, 2023, and 2024)**, and treats the stats from each season as separate features. For example, "Team" is split into "Team_2022", "Team_2023", and "Team_2024".

**In other words, instead of having three rows for three seasons of a player, there is one row per player, with around triple the original number of columns.**

The final dataset has **575 players** and **81 columns** after cleaning.

### **Features used:**

- **General info:** Nation, Age, Position, Team, League, Contract Expiration Date (converted to days left)
- **Playing time:** Matches Played, Starts, Minutes, 90s
- **Output:** Goals, Assists, Yellow Cards, Red Cards
- **Expected stats:** xG, npxG, xAG
- **Progression:** Progressive Carries, Progressive Passes, Progressive Runs
- **Per 90 stats:** Gls.1, Ast.1, G+A.1, G-PK.1, xG.1, xAG.1, npxG.1, npxG+xAG.1

### **Dropped columns:**

- **Player** and **Born** (the age is already there)
- **xG+xAG, npxG+xAG, G+A, G-PK, G+A-PK** (the split metrics are more useful)
- **PK and PKatt** (not heavily weighted for market value)

---

## **Methodology**

1. **Data Cleaning:** fixing number formatting, converting the contract expiration date to days left, label encoding the categorical features, and dropping unneeded columns.
2. **Understanding the Data:** plotting a correlation heatmap against the market value. The most positively correlated features were mostly attacking stats (progressive carries, npxG, xG, goals, goal involvements) along with the days left on the contract. Age was the most negatively correlated, which makes sense since market values usually peak in a player's mid 20s.
3. **Splitting and Scaling:** 80% training, 10% validation, 10% test, scaled with `StandardScaler`.
4. **Training, Validation, and Hyperparameter Tuning:** the first model was heavily overfitting (training loss of 0.0007 vs validation loss of 0.4974), so I tuned the learning rate, weight decay, number of neurons, number of hidden layers, dropout rate, activation function, and normalization until the validation loss stopped improving.
5. **Testing and Error Analysis**

### **Why an MLP?**

- **CNNs** are made for data with spatial patterns like images, not tabular data where the features are mostly independent of each other.
- **Decision Trees/Random Forests** handle non-linear relationships well, but they can overfit if they are too deep, and an MLP gives me a lot more control over the architecture.

---

## **Final Model**

| Hyperparameter | Value |
|---|---|
| Hidden layers | 1 |
| Neurons | 16 |
| Activation | LeakyReLU (0.01) |
| Dropout | 0.2 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Weight decay | None |
| Normalization | None |
| Batch size | 50 |
| Epochs | 1000 |

Interestingly, removing weight decay and normalization gave better results. The reason this might have happened is that I introduced weight decay when the model had more neurons and layers, so once I made the model smaller, the regularization was too harsh and the model started underfitting.

---

## **Results**

| Metric | Value |
|---|---|
| Training loss (MSE, scaled) | 0.0927 |
| Validation loss (MSE, scaled) | 0.3354 |
| Test loss (MSE, scaled) | 0.2066 |
| Test RMSE | ~12 million Euros |
| Average market value | ~20.5 million Euros |

The test loss being lower than the validation loss is a good sign, meaning the model generalizes reasonably well and isn't just memorizing the training data.

However, looking at the **Absolute Value Percent Error** on the test set tells a different story:

- **Q1 (25%):** ~23%
- **Median:** ~57%
- **Q3 (75%):** ~155%
- **Max:** ~742%

22 of the 58 test predictions were off by more than 100%, and some predictions were even negative, which is impossible in real life.

**In short, the model does reasonably well for players with average market values, but fails to generalize for outliers, meaning players with very high or very low market values.** This is most likely because there weren't enough of these players in the training data.

---

## **Future Improvements**

- Expanding the dataset with many more players, especially very high and very low value ones
- Making the model deeper once there's enough data to support it
- Trying BatchNorm and LayerNorm again on a bigger model (they made results worse on this small one)
- Turning the MLP into a multi-task model that could also predict things like injury risk or next season's stats

---

## **How to Run**

The notebook was made in Google Colab. You can either open it there directly or run it locally:

```bash
pip install numpy pandas torch matplotlib seaborn scikit-learn gdown
jupyter notebook market_value_model.ipynb
```

Random seeds are set for NumPy, Python, and PyTorch, and I ran the notebook on a second Colab account to make sure the results were the same.

**Note:** the contract expiration date is converted to "days left" based on today's date, so the exact numbers may shift slightly depending on when you run it.

---

## **Repository Structure**

```
├── market_value_model.ipynb   # main notebook (cleaning, training, evaluation)
├── data/
│   └── merged_player_data.csv # the final dataset
└── README.md
```

---

## **References**

[1] Oles04. (2021). *Market Value Prediction with XGBoost Regressor.* Kaggle. https://www.kaggle.com/code/oles04/market-value-prediction-with-xgboost-regressor

[2] Ozen, U. (2020). *Predicting Football Players' Market Value Using Machine Learning.* Medium. https://yulasozen.medium.com/predicting-football-players-market-value-using-machine-learning-b28be298e91e

Player statistics from [FBref](https://fbref.com/), market values and contract data from [Transfermarkt](https://www.transfermarkt.com/).

Parts of the model creation, training, and testing code are modified versions of the MLP tutorial code provided in the course.
