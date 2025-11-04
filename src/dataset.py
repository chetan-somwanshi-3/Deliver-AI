import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from loguru import logger

def haversine(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance (km) using the Haversine formula."""
    R = 6371  # Earth radius in km
    # np.radians handles NaNs correctly, resulting in NaN for output
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Helper function to clean strings while preserving NaNs
def clean_str_column_for_imputation(series: pd.Series) -> pd.Series:
    """Apply string cleaning only to non-missing values, preserving NaNs."""
    non_null_mask = series.notna()

    # Apply string cleaning only to non-missing values
    series.loc[non_null_mask] = (
        series.loc[non_null_mask].astype(str)
        .str.replace('conditions', '', case=False, regex=False)
        .str.strip().str.title()
    )
    return series

def clean_food_delivery_data_for_imputation(file_path: str) -> pd.DataFrame:
    """
    Clean Food Delivery dataset, preserving NaNs for imputation,
    except for dropping rows with age < 18 and invalid coordinates.
    """

    # ========== LOAD ==========
    df = pd.read_csv(file_path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # ========== BASIC CLEANING ==========
    # Standardize string representations of NaN to np.nan
    df.replace(["NaN ", "NaN", "nan", "conditions NaN", "NaN  ", "NaN "], np.nan, inplace=True)

    # ========== EXTRACT CODES ==========
    df['City_Code'] = df['Delivery_person_ID'].str.extract(r'([A-Z]+)')[0]
    df['Station_Code'] = df['Delivery_person_ID'].str.extract(r'(\d+)')[0]
    df['Agent_Code'] = df['Delivery_person_ID'].str.extract(r'(DEL\d+)')[0]

    # ========== CONVERT DATES & TIMES ==========
    # errors='coerce' turns unparseable values into NaT or NaN, preserving missingness
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

    # ========== CLEAN CATEGORICAL COLUMNS (Preserving NaNs) ==========
    for col in ['Weatherconditions', 'Road_traffic_density', 'Type_of_order', 'Type_of_vehicle', 'Festival', 'City']:
        # Using the modified helper to only clean non-missing strings
        df[col] = clean_str_column_for_imputation(df[col].copy())


    # ========== NUMERIC CLEANING (Using errors='coerce' to preserve NaNs) ==========
    df['multiple_deliveries'] = pd.to_numeric(df['multiple_deliveries'], errors='coerce')
    df['Delivery_person_Age'] = pd.to_numeric(df['Delivery_person_Age'], errors='coerce')
    df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce')

    # Handling 'Time_taken(min)' conversion
    df['Time_taken(min)'] = (
        df['Time_taken(min)'].astype(str)
        .str.replace(r"\(min\)\s*", "", regex=True)
    )
    df['Time_taken(min)'] = pd.to_numeric(df['Time_taken(min)'], errors='coerce') # 'nan' from astype(str) becomes np.nan

    # ========== REMOVE INVALID VALUES (as requested) ==========
    # REMOVED: df = df.dropna(subset=['Delivery_person_Age'])
    
    # Filter out rows where age is less than 18, but KEEP rows where age is NaN
    # The condition is: Keep (NaN) OR (Age >= 18)
    df = df[
        (df['Delivery_person_Age'].isna()) | (df['Delivery_person_Age'] >= 18)
    ].copy()

    # Filter out rows with invalid coordinates (as requested)
    df = df[
        (df['Restaurant_latitude'].between(6.5, 37.1)) &
        (df['Restaurant_longitude'].between(68.7, 97.25)) &
        (df['Delivery_location_latitude'].between(6.5, 37.1)) &
        (df['Delivery_location_longitude'].between(68.7, 97.25))
    ]

    # ========== CALCULATE DISTANCE (Will result in NaN if any coordinate is NaN) ==========
    df['Distance_km'] = haversine(
        df['Restaurant_latitude'], df['Restaurant_longitude'],
        df['Delivery_location_latitude'], df['Delivery_location_longitude']
    )

    # ========== FINAL CLEANUP ==========
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    print(f"✅ Data cleaned successfully: {df.shape[0]} rows remaining")
    return df