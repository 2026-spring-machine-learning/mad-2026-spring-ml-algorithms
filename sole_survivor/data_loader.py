import pandas as pd


FEATURES = [
    "Leadership",
    "MentalToughness",
    "SurvivalSkills",
    "RiskTaking",
    "Resourcefulness",
    "Adaptability",
    "PhysicalFitness",
    "Teamwork",
    "Stubbornness",
]

TARGET = "SurvivalScore"


def load_past_data(path):
    df = pd.read_csv(path)
    df["Name"] = df["Name"].str.strip()
    _check_nulls(df, path)
    return df


def load_next_data(path):
    df = pd.read_csv(path)
    df["Name"] = df["Name"].str.strip()
    _check_nulls(df, path)
    return df


def get_features_and_target(df):
    X = df[FEATURES]
    y = df[TARGET]
    return X, y


def _check_nulls(df, source):
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"WARNING — null values detected in {source}:")
        print(null_counts[null_counts > 0])
