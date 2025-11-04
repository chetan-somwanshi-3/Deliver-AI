import pandas as pd
import numpy as np
from loguru import logger

def haversine(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance (km)."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def clean_str_column_for_imputation(series: pd.Series) -> pd.Series:
    """Clean string values safely (only for non-null entries)."""
    mask = series.notna()
    series.loc[mask] = (
        series.loc[mask].astype(str)
        .str.replace('conditions', '', case=False, regex=False)
        .str.strip()
        .str.title()
    )
    return series


def clean_food_delivery_data_for_imputation(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Standardize text NaN → np.nan
    df.replace(["NaN ", "NaN", "nan", "conditions NaN", "NaN  ", "NaN "], np.nan, inplace=True)

    # Extract ID-based codes
    df['City_Code'] = df['Delivery_person_ID'].str.extract(r'([A-Z]+)')[0]
    df['Station_Code'] = df['Delivery_person_ID'].str.extract(r'(\d+)')[0]
    df['Agent_Code'] = df['Delivery_person_ID'].str.extract(r'(DEL\d+)')[0]

    # Convert dates and times
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce', dayfirst=True)
    df['Time_Orderd'] = pd.to_datetime(df['Time_Orderd'], format='%H:%M:%S', errors='coerce').dt.time

    df['Order_Placed'] = pd.to_datetime(
        df['Order_Date'].astype(str) + ' ' + df['Time_Orderd'].astype(str),
        format='%Y-%m-%d %H:%M:%S', errors='coerce'
    )

    df['Order_Hour'] = df['Order_Placed'].dt.hour
    df['Order_Minute'] = df['Order_Placed'].dt.minute
    df['Order_Time_Category'] = pd.cut(
        df['Order_Hour'],
        bins=[0, 6, 12, 17, 21, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening', 'Late Night'],
        right=False
    )

    # Clean categorical fields
    for col in ['Weatherconditions', 'Road_traffic_density', 'Type_of_order', 'Type_of_vehicle', 'Festival', 'City']:
        df[col] = clean_str_column_for_imputation(df[col].copy())

    # Convert numbers
    df['multiple_deliveries'] = pd.to_numeric(df['multiple_deliveries'], errors='coerce')
    df['Delivery_person_Age'] = pd.to_numeric(df['Delivery_person_Age'], errors='coerce')
    df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce')

    df['Time_taken(min)'] = (
        df['Time_taken(min)'].astype(str)
        .str.replace(r"\(min\)\s*", "", regex=True)
    )
    df['Time_taken(min)'] = pd.to_numeric(df['Time_taken(min)'], errors='coerce')

    # Remove unrealistic ages
    df.loc[(df['Delivery_person_Age'] < 18) | (df['Delivery_person_Age'] > 70), 'Delivery_person_Age'] = np.nan

    # -------------------------
    # ✅ No Coordinate Imputation
    # -------------------------
    coord_cols = [
        'Restaurant_latitude', 'Restaurant_longitude',
        'Delivery_location_latitude', 'Delivery_location_longitude'
    ]

    # Convert to numeric
    df[coord_cols] = df[coord_cols].apply(pd.to_numeric, errors='coerce')

    # Compute Distance only where coordinates are present
    df['Distance_km'] = np.where(
        df[coord_cols].notna().all(axis=1),
        haversine(
            df['Restaurant_latitude'], df['Restaurant_longitude'],
            df['Delivery_location_latitude'], df['Delivery_location_longitude']
        ),
        np.nan
    )

    # Remove duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    print(f"✅ Data cleaned successfully: {df.shape[0]} rows remaining")
    print(f"📦 Missing Distance values: {df['Distance_km'].isna().sum()}")

    return df

