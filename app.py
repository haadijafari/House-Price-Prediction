"""Tehran Real Estate Price Valuation Web Application.

An interactive Streamlit application powered by a trained Scikit-Learn ensemble model.
Estimates intrinsic property values in USD and dynamically projects real-time valuations
in Iranian Tomans based on configurable exchange rates to hedge against currency inflation.
"""

import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# 1. CLASS DEFINITION FOR PIPELINE UNPICKLING
# ==============================================================================


class RealEstateFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer computing living space ratios and composite amenity scores."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "RealEstateFeatureEngineer":
        """Fit method (no-op as calculations are deterministic)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Derive domain features: Area_per_Room and Total_Amenities."""
        X_out = X.copy()
        rooms_clipped = X_out["Room"].clip(lower=1)
        X_out["Area_per_Room"] = X_out["Area"] / rooms_clipped

        amenity_cols = [col for col in ["Parking", "Warehouse", "Elevator"] if col in X_out.columns]
        X_out["Total_Amenities"] = X_out[amenity_cols].sum(axis=1)
        return X_out


# Register class in __main__ to ensure seamless unpickling
sys.modules["__main__"].RealEstateFeatureEngineer = RealEstateFeatureEngineer

# ==============================================================================
# 2. PAGE CONFIGURATION & DATA CACHING
# ==============================================================================

load_dotenv()
DEFAULT_EXCHANGE_RATE: float = float(os.getenv("USD_TO_TOMAN_RATE", "60000"))

st.set_page_config(
    page_title="Tehran Real Estate Valuation System",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model_pipeline(model_path: str = "model.joblib"):
    """Load serialized scikit-learn model pipeline."""
    path = Path(model_path)
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_neighborhood_data(
    data_path: str = "data/housePrice.csv",
) -> tuple[list[str], dict[str, float]]:
    """Extract sorted neighborhood names and historical benchmark price/m² in USD."""
    path = Path(data_path)
    if not path.exists():
        return [], {}

    df = pd.read_csv(path)
    df = df.dropna(subset=["Address", "Price(USD)", "Area"])

    # Clean Area
    df["Area"] = pd.to_numeric(
        df["Area"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    df = df[(df["Area"] >= 15) & (df["Area"] <= 1000) & (df["Price(USD)"] > 0)]

    df["USD_per_sqm"] = df["Price(USD)"] / df["Area"]
    addresses = sorted(df["Address"].astype(str).str.strip().unique().tolist())
    median_prices = df.groupby("Address")["USD_per_sqm"].median().to_dict()

    return addresses, median_prices


# Load cached model & data
model = load_model_pipeline()
addresses, neighborhood_medians = load_neighborhood_data()

# Fallback addresses if dataset file is absent
if not addresses:
    addresses = [
        "Amirabad",
        "Farmanieh",
        "Gheitarieh",
        "Jordan",
        "Niavaran",
        "Punak",
        "Saadat Abad",
        "Shahran",
        "Tajrish",
        "Velenjak",
        "Zaferanieh",
    ]

# ==============================================================================
# 3. SIDEBAR CONTROLS
# ==============================================================================

with st.sidebar:
    st.header("⚙️ Property Parameters")

    selected_address = st.selectbox(
        "📍 Neighborhood / District",
        options=addresses,
        index=addresses.index("Saadat Abad") if "Saadat Abad" in addresses else 0,
        help="Select a Tehran residential district.",
    )

    area = st.number_input(
        "📐 Area (Square Meters)",
        min_value=15.0,
        max_value=1000.0,
        value=100.0,
        step=5.0,
        help="Usable interior living area in m².",
    )

    rooms = st.slider(
        "🛏️ Bedrooms",
        min_value=0,
        max_value=5,
        value=2,
        help="Number of bedrooms (0 for studio apartments).",
    )

    st.subheader("✨ Amenities")
    has_parking = st.checkbox("🚗 Parking Space", value=True)
    has_warehouse = st.checkbox("📦 Storage Warehouse", value=True)
    has_elevator = st.checkbox("🛗 Building Elevator", value=True)

    st.markdown("---")
    st.header("💱 Currency & Inflation Control")
    st.info(
        "In Iranian real estate, intrinsic valuation is benchmarked in USD to hedge against hyperinflation. "
        "Adjust this rate to reflect current market rates."
    )

    exchange_rate = st.number_input(
        "Current USD / Toman Exchange Rate",
        min_value=1000.0,
        # max_value=500000.0,
        value=DEFAULT_EXCHANGE_RATE,
        step=1000.0,
        help="Adjust exchange rate for live Toman price projection.",
    )

# ==============================================================================
# 4. MAIN DASHBOARD & VALUATION DISPLAY
# ==============================================================================

st.title("🏡 Tehran Real Estate Valuation System")
st.caption(
    "Predictive property valuation powered by Machine Learning. "
    "Features intrinsic USD valuation and dynamic currency conversion to insulate estimates from currency inflation."
)

st.markdown("---")

if model is None:
    st.error(
        "⚠️ Model artifact (`model.joblib`) not found! "
        "Please run the `House_Price.ipynb` notebook to train and serialize the model pipeline."
    )
else:
    # Construct input dataframe
    input_data = pd.DataFrame(
        [
            {
                "Area": float(area),
                "Room": int(rooms),
                "Parking": int(has_parking),
                "Warehouse": int(has_warehouse),
                "Elevator": int(has_elevator),
                "Address": selected_address.strip(),
            }
        ]
    )

    # Perform prediction
    predicted_usd = float(model.predict(input_data)[0])
    predicted_toman = predicted_usd * exchange_rate
    predicted_billion_toman = predicted_toman / 1e9
    usd_per_sqm = predicted_usd / float(area)
    toman_per_sqm = predicted_toman / float(area)

    # Key Metrics Display
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="💵 Intrinsic USD Value",
            value=f"${predicted_usd:,.0f} USD",
            help="Estimated intrinsic property purchasing power in USD.",
        )

    with col2:
        st.metric(
            label="🇮🇷 Projected Toman Value",
            value=f"{predicted_billion_toman:.2f} Billion",
            delta=f"{predicted_toman:,.0f} Tomans",
            delta_color="off",
            help=f"Projected price at {exchange_rate:,.0f} Tomans/USD.",
        )

    with col3:
        st.metric(
            label="📐 USD Price per m²",
            value=f"${usd_per_sqm:,.1f}/m²",
        )

    with col4:
        st.metric(
            label="📊 Toman Price per m²",
            value=f"{toman_per_sqm / 1e6:.1f}M Tomans/m²",
            delta=f"{toman_per_sqm:,.0f} Tomans",
            delta_color="off",
        )

    st.markdown("---")

    # Neighborhood Comparison & Context Section
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📋 Property Specification Summary")
        spec_df = pd.DataFrame(
            {
                "Attribute": [
                    "District",
                    "Area",
                    "Bedrooms",
                    "Parking",
                    "Warehouse",
                    "Elevator",
                    "Exchange Rate",
                ],
                "Value": [
                    selected_address,
                    f"{area} m²",
                    f"{rooms} Bedrooms",
                    "Included" if has_parking else "None",
                    "Included" if has_warehouse else "None",
                    "Included" if has_elevator else "None",
                    f"{exchange_rate:,.0f} Tomans/USD",
                ],
            }
        )
        st.table(spec_df)

    with col_right:
        st.subheader("📍 Neighborhood Market Benchmark")
        neighborhood_median = neighborhood_medians.get(selected_address)

        if neighborhood_median:
            median_toman_sqm = neighborhood_median * exchange_rate
            diff_pct = ((usd_per_sqm - neighborhood_median) / neighborhood_median) * 100

            st.write(
                f"**District:** {selected_address}\n\n"
                f"- **District Benchmark Median:** ${neighborhood_median:,.1f}/m² ({median_toman_sqm / 1e6:.1f}M Tomans/m²)\n"
                f"- **Property vs. Benchmark:** `{diff_pct:+.1f}%`"
            )

            if diff_pct > 5:
                st.info(
                    "💎 This property is estimated at a premium relative to the district median (often driven by larger area, high bedroom count, or full amenities)."
                )
            elif diff_pct < -5:
                st.success(
                    "🏷️ This property is estimated below the district median per square meter, indicating an attractive relative valuation."
                )
            else:
                st.info(
                    "⚖️ This property aligns closely with the district's prevailing median valuation."
                )
        else:
            st.write("Historical benchmark statistics for this specific district are limited.")

st.markdown("---")
st.caption("Developed by Haadi Jafari | Tehran Real Estate Valuation Project")
