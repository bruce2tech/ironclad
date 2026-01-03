import math
import pandas as pd

file_path = "Admission_Predict_Ver1.1.csv"
df = pd.read_csv(file_path)

# Drop the serial column if present
cols = [c for c in df.columns if c.lower().strip() not in {"serial no.", "serial no"}]
X = df[cols].copy()

# --- Helpers (no numpy/statistics) ---

def mean(xs):
    n = len(xs)
    return sum(xs) / n

def stddev(xs, ddof=1):
    n = len(xs)
    if n - ddof <= 0:
        return float("nan")
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - ddof)  # sample variance
    return math.sqrt(var)

def covariance(xs, ys, ddof=1):
    n = len(xs)
    if n != len(ys) or n - ddof <= 0:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - ddof)

def pearsonr(xs, ys):
    # Pairwise compute with sample stats
    sx = stddev(xs, ddof=1)
    sy = stddev(ys, ddof=1)
    if sx == 0 or sy == 0 or math.isnan(sx) or math.isnan(sy):
        return float("nan")
    cov = covariance(xs, ys, ddof=1)
    return cov / (sx * sy)

# --- Standard deviations per column (sample) ---

std_by_col = {}
for col in X.columns:
    # Drop NaNs for this column
    vals = [float(v) for v in X[col].tolist() if pd.notna(v)]
    std_by_col[col] = stddev(vals, ddof=1)

std_dev_series = pd.Series(std_by_col, name="std_sample")
print(std_dev_series)

# --- Correlation matrix (Pearson) from scratch ---

# We’ll do pairwise dropna per column pair
corr = pd.DataFrame(index=X.columns, columns=X.columns, dtype=float)

for i, c1 in enumerate(X.columns):
    for j, c2 in enumerate(X.columns):
        if j < i:
            corr.loc[c1, c2] = corr.loc[c2, c1]  # symmetry
            continue

        # Pairwise non-NaN rows
        pairs = [(x, y) for x, y in zip(X[c1].tolist(), X[c2].tolist())
                 if pd.notna(x) and pd.notna(y)]
        if len(pairs) < 2:
            r = float("nan")
        else:
            xs, ys = zip(*pairs)
            xs = [float(x) for x in xs]
            ys = [float(y) for y in ys]
            r = pearsonr(xs, ys)

        corr.loc[c1, c2] = r
        corr.loc[c2, c1] = r

print("\nPearson correlation (from scratch, sample-based):")
print(corr)