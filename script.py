import json
import math
from pathlib import Path

import numpy as np
import xarray as xr

# 1. Open the age grid (6 arc-min GTS2012)
nc_path = "age.2020.1.GTS2012.6m.nc"   # change if your filename is different
ds = xr.open_dataset(nc_path)

# Use the first data variable (or explicitly ds["age"] if you know its name)
var_name = list(ds.data_vars)[0]
da = ds[var_name]

# 2. Define the South Atlantic box
lon_min, lon_max = -70, 20   # W to E
lat_min, lat_max = -60, 20   # S to N

# 3. Robust crop that handles either lat direction
lat = da["lat"]
if float(lat[0]) > float(lat[-1]):
    # lat decreases from north to south (e.g. 90 -> -90)
    da_reg = da.sel(lon=slice(lon_min, lon_max),
                    lat=slice(lat_max, lat_min))
else:
    # lat increases from south to north (e.g. -90 -> 90)
    da_reg = da.sel(lon=slice(lon_min, lon_max),
                    lat=slice(lat_min, lat_max))

print("Cropped sizes (lon, lat):", da_reg.lon.size, da_reg.lat.size)
if da_reg.lon.size == 0 or da_reg.lat.size == 0:
    raise RuntimeError("Cropped region is empty; check lon/lat bounds.")

# 4. Convert to numpy and clean NaN/Inf -> None
vals = da_reg.to_numpy()  # 2D numpy array
data_clean = []
nan_count = 0

for row in vals:
    clean_row = []
    for v in row:
        # Handle masked / None
        if v is None:
            clean_row.append(None)
            continue

        # Convert to float and check finiteness
        try:
            f = float(v)
        except Exception:
            clean_row.append(None)
            nan_count += 1
            continue

        if math.isfinite(f):
            clean_row.append(f)
        else:
            clean_row.append(None)
            nan_count += 1

    data_clean.append(clean_row)

print("Replaced", nan_count, "non-finite values with None")

# 5. Build JSON object
out = {
    "var_name": var_name,
    "units": da.attrs.get("units", ""),
    "lon": da_reg["lon"].values.tolist(),
    "lat": da_reg["lat"].values.tolist(),
    "data": data_clean,  # 2D [lat_index][lon_index]
}

json_path = Path("age_SAtl_6m.json")
with json_path.open("w", encoding="utf-8") as f:
    # Forbid NaN/Infinity: will raise ValueError if any survive
    json.dump(out, f, allow_nan=False)

print("Wrote JSON to:", json_path.resolve())

# 6. Extra safety check: ensure no 'NaN' token in text
text = json_path.read_text(encoding="utf-8")
if "NaN" in text or "nan" in text:
    print("WARNING: 'NaN' text still found in JSON.")
else:
    print("Confirmed: no 'NaN' tokens in JSON.")
