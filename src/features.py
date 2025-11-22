import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Performs Feature Engineering on the cleaned dataset:
    1. Calculates Haversine Distance (spatial).
    2. Extracts Temporal Features (day of week, weekend).
    3. Encodes Categorical Variables (Label & One-Hot).
    """
    
    def __init__(self):
        # Map ordinal variables to numbers manually to preserve rank
        self.traffic_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Jam': 4}
        self.city_map = {'Semi-Urban': 1, 'Urban': 2, 'Metropolitian': 3}
        
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculates distance in km between two lat/lon coordinates.
        """
        R = 6371  # Earth radius in kilometers
        
        # Convert degrees to radians
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        
        a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c

    def fit(self, X, y=None):
        return self  # Nothing to learn for simple engineering

    def transform(self, X):
        df = X.copy()
        
        # --- 1. Spatial Features (Haversine) ---
        # Ensure we have the coordinates
        if {'Restaurant_latitude', 'Restaurant_longitude', 
            'Delivery_location_latitude', 'Delivery_location_longitude'}.issubset(df.columns):
            
            df['distance_km'] = self.haversine_distance(
                df['Restaurant_latitude'], df['Restaurant_longitude'],
                df['Delivery_location_latitude'], df['Delivery_location_longitude']
            )
            
        # --- 2. Temporal Features ---
        if 'Order_Date' in df.columns:
            df['day_of_week'] = df['Order_Date'].dt.dayofweek  # 0=Mon, 6=Sun
            df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
            df['month'] = df['Order_Date'].dt.month
            
            # We can drop the original date col now as it's not model-friendly
            df.drop(columns=['Order_Date'], inplace=True)

        # --- 3. Categorical Encoding ---
        
        # A. Ordinal Encoding (Rank matters)
        if 'Road_traffic_density' in df.columns:
            df['Road_traffic_density'] = df['Road_traffic_density'].map(self.traffic_map).fillna(0)
            
        if 'City' in df.columns:
            df['City'] = df['City'].map(self.city_map).fillna(0)
            
        # B. Binary Encoding
        if 'Festival' in df.columns:
            df['Festival'] = df['Festival'].apply(lambda x: 1 if str(x).strip() == 'Yes' else 0)

        # C. One-Hot Encoding (Nominal variables)
        # Note: In a strict pipeline, use sklearn OneHotEncoder. 
        # For this example, we use get_dummies for readability.
        cols_to_dummy = ['Weatherconditions', 'Type_of_order', 'Type_of_vehicle', 'Order_Time_Category']
        cols_present = [c for c in cols_to_dummy if c in df.columns]
        
        df = pd.get_dummies(df, columns=cols_present, drop_first=True)
        
        # Drop IDs and other non-feature columns
        drop_cols = ['ID', 'Delivery_person_ID', 'Time_Orderd', 'Time_Order_picked']
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
        
        return df