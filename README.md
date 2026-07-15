# Used Car Price Predictor

Predicts a fair market price for a used car based on its specs (mileage, age, engine size, brand, fuel type, transmission, etc.) so you have a number to negotiate against instead of just trusting the sticker price.

I started this one mostly to practice the full regression workflow end to end — cleaning, feature engineering, comparing multiple models honestly instead of just picking one — using a public 50,000-row used car dataset (25 columns: brand, model, year, mileage, engine size, horsepower, accident history, insurance status, etc.). Worth being upfront that this dataset is from Kaggle, not something I scraped myself — unlike my [apartment fairness tool](../ApartmentScraperProject), which uses real listings I collected.

## What's in the notebook

`EDA&Modeling_CarProject.ipynb` walks through:

- Cleaning: dropping duplicates, filling missing mileage/engine size with the median
- Feature engineering: `CarAge` from `Year`, binning mileage into 5 buckets (Very Low → Very High), one-hot encoding brand/fuel type/transmission
- A quick EDA pass (mileage vs. price scatterplot, etc.)
- Two models, on purpose, so I could see how much a simple baseline actually loses to a more flexible model

## Results

| Model | MAE | R² |
|---|---|---|
| Linear Regression | $5,810 | 0.716 |
| Random Forest (200 trees) | **$641** | **0.989** |

The gap is bigger than I expected going in. Linear regression assumes price moves in a straight line with each feature, but car depreciation isn't linear — a car losing its first 20k miles hits price very differently than one going from 100k to 120k, and brand/mileage/age interact with each other in ways a single linear coefficient can't capture. Random Forest picks all of that up for free, which is basically the whole lesson of this project.

Final predictions (plus the difference between actual and predicted price, in case a listing is over/under market) get exported to `used_cars_with_predictions.csv` — I originally built that for a Power BI dashboard on top of this data.

## Stack

Python · pandas · numpy · scikit-learn · matplotlib · seaborn · Jupyter

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook "EDA&Modeling_CarProject.ipynb"
```

## What's next

- Wrap the Random Forest model in a small Streamlit app (like my other projects) so you can plug in a car's specs and get a price instead of reading it off a notebook
- Try XGBoost / hyperparameter tuning to see if there's more room past Random Forest's default settings
- Add a "confidence range" instead of a single point estimate — a used car's fair price is really a range, not one number
