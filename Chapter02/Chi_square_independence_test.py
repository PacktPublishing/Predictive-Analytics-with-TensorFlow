import pandas as pd
from scipy import stats

survey = pd.read_csv("data/survey.csv")
# Tabulating 2 variables with row & column variables respectively
survey_tab = pd.crosstab(survey.Smoke, survey.Exer, margins = True)
# Creating observed table for analysis
observed = survey_tab.ix[0:4,0:3]

contg = stats.chi2_contingency(observed= observed)
p_value = round(contg[1],3)
print ("P-value is: ",p_value)
