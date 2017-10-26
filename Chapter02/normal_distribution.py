from scipy import stats

xbar = 67
mu0 = 52
s = 16.3

# Compute z-score
z = (67-52)/16.3

# Calculating probability under the curve
p_val = 1 - stats.norm.cdf(z)
print("Prob. to score more than 67 is ", round(p_val*100, 2), "%")
