from scipy.stats import chisquare

x = [209, 280, 225, 248]
chi_statistic, p_value = chisquare(x)
print(chi_statistic)
print(p_value)
