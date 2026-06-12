import data_loader as dl
import pandas as pd

panel = dl.load_price_panel()

# How many stocks are present (non-NaN) on each date?
coverage = panel.notna().sum(axis=1)
print("Max stocks present on any single date:", coverage.max(), "of", panel.shape[1])
print("Dates with all 50 present:", (coverage == panel.shape[1]).sum())

# Per-stock: first valid date and total missing days within its own lifespan
print("\nPer-stock start dates (latest 8):")
starts = panel.apply(lambda c: c.first_valid_index())
print(starts.sort_values(ascending=False).head(8))

# Which stocks have internal gaps (NaNs AFTER they start trading)?
print("\nStocks with internal gaps (NaNs after first listing):")
for t in panel.columns:
    s = panel[t]
    fv = s.first_valid_index()
    if fv is not None:
        internal_na = s.loc[fv:].isna().sum()
        if internal_na > 0:
            print(f"  {t}: {internal_na} internal missing days")