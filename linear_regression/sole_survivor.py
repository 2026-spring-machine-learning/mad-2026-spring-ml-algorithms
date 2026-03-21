import numpy as np
import pandas as pd
import sklearn.linear_model as lm

# Add plotting imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import seaborn as sns


##step 1 - checking the results of the sole_survivor_past.csv 
    #  Independent Variables = Leadership, MentalToughness, SurvivalSkills, Risk,Taking, Resourcefullness, Adaptability, PhysicalFitness, Teamwork
    # Dependant Variable "SurvivalScore"

def main():
    survivor_past = pd.read_csv("H:/MATC/3_2026 Spring Sem/MachineLearning/mad-2026-spring-ml-algorithms/linear_regression/sole_survivor_past.csv")
    print(survivor_past)
    # predict(cars_df)


if __name__ == "__main__":
    main()
