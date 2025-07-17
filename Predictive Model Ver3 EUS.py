import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import io

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import calendar

# Try to import holidays package
try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Integrated Demand Predictor",
    page_icon="🔮",
    layout="wide"
)

st.markdown("""
# 🔮 Integrated Demand Prediction Tool with Prebooking Analysis
### Predict Assisted Travel Demand with Real-time Prebooking Adjustments
*Combines seasonal patterns, UK holidays, and live prebooking intelligence*
""")

# ============================================================================
# PREBOOKING ANALYZER CLASS
# ============================================================================

class PrebookingAnalyzer:
    """Analyses relationship between prebookings and final demand"""
    
    def __init__(self):
        self.prebooking_models = {}
        self.prebooking_stats = {}
        self.day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    def analyze_prebooking_patterns(self, historical_data):
        """Analyse prebooking patterns from historical data"""
        
        # Calculate days in advance for each booking
        historical_data['days_in_advance'] = (
            historical_data['scheduled_departure_date'] - 
            historical_data['booking_created_date']
        ).dt.days
        
        # Filter valid bookings
        historical_data = historical_data[
            (historical_data['days_in_advance'] >= 0) & 
            (historical_data['days_in_advance'] <= 365)
        ]
        
        # Group by departure date to get cumulative bookings
        prebooking_profiles = []
        
        for departure_date in historical_data['scheduled_departure_date'].dt.date.unique():
            date_bookings = historical_data[
                historical_data['scheduled_departure_date'].dt.date == departure_date
            ]
            
            if len(date_bookings) < 10:
                continue
            
            total_bookings = len(date_bookings)
            day_of_week = pd.to_datetime(departure_date).dayofweek
            
            for days_before in range(0, 8):  # 0-7 days before
                bookings_by_cutoff = date_bookings[
                    date_bookings['days_in_advance'] >= days_before
                ]
                count_at_cutoff = len(bookings_by_cutoff)
                percentage_at_cutoff = count_at_cutoff / total_bookings if total_bookings > 0 else 0
                
                prebooking_profiles.append({
                    'departure_date': departure_date,
                    'day_of_week': day_of_week,
                    'days_before': days_before,
                    'bookings_at_cutoff': count_at_cutoff,
                    'total_bookings': total_bookings,
                    'percentage_booked': percentage_at_cutoff
                })
        
        prebooking_df = pd.DataFrame(prebooking_profiles)
        
        # Analyze patterns by day of week
        for dow in range(7):
            dow_data = prebooking_df[prebooking_df['day_of_week'] == dow]
            
            if len(dow_data) < 10:
                continue
            
            stats_by_days = {}
            
            for days_before in range(0, 8):
                cutoff_data = dow_data[dow_data['days_before'] == days_before]
                
                if len(cutoff_data) > 0:
                    stats_by_days[days_before] = {
                        'mean_percentage': cutoff_data['percentage_booked'].mean(),
                        'std_percentage': cutoff_data['percentage_booked'].std(),
                        'median_percentage': cutoff_data['percentage_booked'].median(),
                        'q25_percentage': cutoff_data['percentage_booked'].quantile(0.25),
                        'q75_percentage': cutoff_data['percentage_booked'].quantile(0.75),
                        'sample_size': len(cutoff_data)
                    }
            
            self.prebooking_stats[dow] = stats_by_days
        
        return prebooking_df
    
    def calculate_prebooking_adjustment(self, day_of_week, current_prebookings, 
                                      base_prediction, days_until_departure):
        """Calculate adjustment factor based on prebooking levels"""
        
        if day_of_week not in self.prebooking_stats:
            return 1.0, "No prebooking model available", None
        
        stats = self.prebooking_stats[day_of_week].get(days_until_departure, {})
        
        if not stats or stats.get('sample_size', 0) < 10:
            return 1.0, "Insufficient historical data", None
        
        expected_percentage = stats['mean_percentage']
        expected_prebookings = base_prediction * expected_percentage
        
        if expected_percentage > 0:
            predicted_final_demand = current_prebookings / expected_percentage
        else:
            predicted_final_demand = base_prediction
        
        if base_prediction > 0:
            adjustment_ratio = predicted_final_demand / base_prediction
        else:
            adjustment_ratio = 1.0
        
        analysis = {
            'expected_prebookings': expected_prebookings,
            'actual_prebookings': current_prebookings,
            'expected_percentage': expected_percentage,
            'predicted_final_demand': predicted_final_demand,
            'base_prediction': base_prediction,
            'adjustment_ratio': adjustment_ratio
        }
        
        if adjustment_ratio > 1.0:
            explanation = f"Prebookings {current_prebookings:.0f} vs {expected_prebookings:.0f} expected → Predicting {predicted_final_demand:.0f} total"
        elif adjustment_ratio < 1.0:
            explanation = f"Prebookings {current_prebookings:.0f} vs {expected_prebookings:.0f} expected → Lower demand signal"
        else:
            explanation = "Prebookings match expectations"
        
        return adjustment_ratio, explanation, analysis

# ============================================================================
# HOLIDAY PATTERN CLASS
# ============================================================================

class AssistedTravelHolidayPatterns:
    """Handles UK bank holiday patterns specific to assisted travel customers"""
    
    def __init__(self):
        current_year = datetime.now().year
        if HOLIDAYS_AVAILABLE:
            self.uk_holidays = holidays.UK(years=range(current_year-2, current_year+2))
        else:
            self.uk_holidays = self._get_manual_holidays()
        
        self.pattern_templates = {
            'easter': {
                'good_friday': {
                    -2: 1.3, -1: 1.35, 0: 1.3, 1: 0.85, 2: 0.8
                },
                'easter_monday': {
                    -2: 0.8, -1: 0.85, 0: 1.4, 1: 1.35, 2: 1.2
                }
            },
            'standard_monday': {
                -3: 1.2, -2: 1.25, -1: 1.0, 0: 0.9, 1: 1.4, 2: 1.25
            },
            'christmas': {
                -3: 1.35, -2: 1.4, -1: 1.45, 0: 0.7, 1: 0.8, 2: 1.1, 3: 1.2
            },
            'new_year': {
                -2: 1.2, -1: 1.25, 0: 0.8, 1: 1.1, 2: 1.3
            }
        }
    
    def _get_manual_holidays(self):
        """Manual holiday definitions as fallback"""
        current_year = datetime.now().year
        holidays_dict = {}
        
        for year in [current_year, current_year + 1]:
            holidays_dict.update({
                pd.Timestamp(f'{year}-01-01'): "New Year's Day",
                pd.Timestamp(f'{year}-12-25'): 'Christmas Day',
                pd.Timestamp(f'{year}-12-26'): 'Boxing Day'
            })
            
            if year == 2025:
                holidays_dict.update({
                    pd.Timestamp('2025-04-18'): 'Good Friday',
                    pd.Timestamp('2025-04-21'): 'Easter Monday',
                })
            elif year == 2026:
                holidays_dict.update({
                    pd.Timestamp('2026-04-03'): 'Good Friday',
                    pd.Timestamp('2026-04-06'): 'Easter Monday',
                })
        
        return holidays_dict
    
    def get_holiday_factor(self, check_date, base_prediction=None):
        """Get the appropriate multiplier for a given date"""
        
        if not isinstance(check_date, pd.Timestamp):
            check_date = pd.Timestamp(check_date)
        
        best_factor = 1.0
        holiday_name = None
        
        for holiday_date, name in self.uk_holidays.items():
            if not isinstance(holiday_date, pd.Timestamp):
                holiday_date = pd.Timestamp(holiday_date)
                
            days_diff = (check_date - holiday_date).days
            
            if abs(days_diff) > 5:
                continue
                
            pattern = self.get_pattern_for_holiday(name, holiday_date)
            
            if days_diff in pattern:
                factor = pattern[days_diff]
                
                # Adjust for day of week preferences
                if check_date.weekday() in [5, 6]:  # Weekend
                    factor *= 0.9
                elif check_date.weekday() in [1, 2, 3]:  # Tue-Thu
                    factor *= 1.05
                
                if factor != 1.0 and abs(factor - 1.0) > abs(best_factor - 1.0):
                    best_factor = factor
                    holiday_name = name
        
        return best_factor, holiday_name
    
    def get_pattern_for_holiday(self, holiday_name, holiday_date):
        """Match holiday to appropriate pattern template"""
        
        holiday_name = str(holiday_name)
        
        if "Good Friday" in holiday_name:
            return self.pattern_templates['easter']['good_friday']
        elif "Easter Monday" in holiday_name:
            return self.pattern_templates['easter']['easter_monday']
        elif "Christmas" in holiday_name:
            return self.pattern_templates['christmas']
        elif "New Year" in holiday_name:
            return self.pattern_templates['new_year']
        elif holiday_date.weekday() == 0:  # Monday holiday
            return self.pattern_templates['standard_monday']
        else:
            return {-2: 1.2, -1: 1.25, 0: 1.1, 1: 1.25, 2: 1.2}

# ============================================================================
# DATA LOADING - SIMPLE MANUAL UPLOAD ONLY
# ============================================================================

@st.cache_data(show_spinner=False)
def load_and_process_data():
    """Load data from manual file upload"""
    
    st.markdown("""
    ### 📁 Upload Data Files
    Please upload your CSV files to begin.
    """)
    
    uploaded_files = st.file_uploader(
        "Choose CSV files",
        type=['csv'],
        accept_multiple_files=True,
        help="Select 2023, 2024, and 2025 Database files"
    )
    
    if not uploaded_files:
        st.info("Please upload at least 2 years of data (2023, 2024, and/or 2025 Database.csv)")
        st.stop()
    
    # Process uploaded files
    yearly_data = {}
    has_prebooking_data = False
    
    for uploaded_file in uploaded_files:
        try:
            # Extract year from filename
            filename = uploaded_file.name
            year = None
            for y in [2023, 2024, 2025]:
                if str(y) in filename:
                    year = y
                    break
            
            if year is None:
                st.warning(f"Could not identify year from filename: {filename}")
                continue
            
            # Read CSV
            df = pd.read_csv(uploaded_file, low_memory=False)
            df_euston = df[df['station_code'] == "EUS"].copy()
            
            if 'booking_created_date' in df_euston.columns:
                has_prebooking_data = True
            
            if len(df_euston) > 0:
                yearly_data[year] = df_euston
                st.success(f"✅ Loaded {year} data: {len(df_euston):,} records")
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    if len(yearly_data) < 2:
        st.error("Need at least 2 years of data to generate predictions")
        st.stop()
    
    # Process each year
    processed_years = {}
    
    for year, df in yearly_data.items():
        # Convert dates
        df['scheduled_departure_date'] = pd.to_datetime(
            df['scheduled_departure_date'], 
            dayfirst=True,
            errors='coerce'
        )
        
        if 'booking_created_date' in df.columns:
            df['booking_created_date'] = pd.to_datetime(
                df['booking_created_date'],
                dayfirst=True,
                errors='coerce'
            )
        
        df = df.dropna(subset=['scheduled_departure_date'])
        
        # Create daily bookings
        daily_bookings = df.groupby('scheduled_departure_date').size().reset_index(name='bookings')
        
        # Full year date range
        year_start = pd.Timestamp(f'{year}-01-01')
        year_end = pd.Timestamp(f'{year}-12-31')
        
        if daily_bookings['scheduled_departure_date'].min() > year_start:
            year_start = daily_bookings['scheduled_departure_date'].min()
        if daily_bookings['scheduled_departure_date'].max() < year_end:
            year_end = daily_bookings['scheduled_departure_date'].max()
        
        full_dates = pd.date_range(start=year_start, end=year_end, freq='D')
        full_dates_df = pd.DataFrame({'scheduled_departure_date': full_dates})
        
        yearly_bookings = full_dates_df.merge(daily_bookings, on='scheduled_departure_date', how='left')
        yearly_bookings['bookings'] = yearly_bookings['bookings'].fillna(0)
        yearly_bookings['year'] = year
        yearly_bookings['day_of_year'] = yearly_bookings['scheduled_departure_date'].dt.dayofyear
        yearly_bookings['day_of_week'] = yearly_bookings['scheduled_departure_date'].dt.dayofweek
        yearly_bookings['month'] = yearly_bookings['scheduled_departure_date'].dt.month
        yearly_bookings['week_of_year'] = yearly_bookings['scheduled_departure_date'].dt.isocalendar().week
        
        processed_years[year] = yearly_bookings
    
    # Combine raw data if prebooking analysis needed
    all_raw_data = None
    if has_prebooking_data:
        all_raw_data = pd.concat([yearly_data[year] for year in yearly_data.keys()])
    
    return processed_years, all_raw_data, has_prebooking_data

# ============================================================================
# MODEL TRAINING AND PREDICTION
# ============================================================================

@st.cache_resource(show_spinner=False)
def train_prediction_model(processed_years):
    """Train the base model on all historical data"""
    
    all_historical = pd.concat(processed_years.values())
    
    # Calculate growth trend
    yearly_totals = []
    for year, data in processed_years.items():
        total = data['bookings'].sum()
        yearly_totals.append((year, total))
    
    years = [x[0] for x in yearly_totals]
    totals = [x[1] for x in yearly_totals]
    
    # Calculate growth factor
    if len(years) >= 2:
        growth_rates = []
        for i in range(1, len(yearly_totals)):
            prev_total = yearly_totals[i-1][1]
            curr_total = yearly_totals[i][1]
            growth = (curr_total - prev_total) / prev_total if prev_total > 0 else 0
            growth_rates.append(growth)
        
        if len(growth_rates) > 1:
            weights = [0.7 ** (len(growth_rates) - i - 1) for i in range(len(growth_rates))]
            weights = [w / sum(weights) for w in weights]
            avg_growth = sum(g * w for g, w in zip(growth_rates, weights))
        else:
            avg_growth = growth_rates[0]
        
        years_ahead = datetime.now().year - years[-1]
        growth_factor = (1 + avg_growth) ** years_ahead
    else:
        growth_factor = 1.1
    
    # Calculate seasonal baseline
    seasonal_baseline = all_historical.groupby('day_of_year')['bookings'].median().to_dict()
    
    # Fill missing days
    all_days = range(1, 367)
    for day in all_days:
        if day not in seasonal_baseline:
            nearby_days = [d for d in seasonal_baseline.keys() if abs(d - day) <= 7]
            if nearby_days:
                seasonal_baseline[day] = np.mean([seasonal_baseline[d] for d in nearby_days])
            else:
                seasonal_baseline[day] = all_historical['bookings'].median()
    
    # Calculate factors
    dow_medians = all_historical.groupby('day_of_week')['bookings'].median()
    overall_median = all_historical['bookings'].median()
    dow_factors = (dow_medians / overall_median).to_dict() if overall_median > 0 else {}
    
    monthly_medians = all_historical.groupby('month')['bookings'].median()
    monthly_factors = (monthly_medians / overall_median).to_dict() if overall_median > 0 else {}
    
    daily_std = all_historical.groupby('day_of_year')['bookings'].std().to_dict()
    
    return {
        'seasonal_baseline': seasonal_baseline,
        'dow_factors': dow_factors,
        'monthly_factors': monthly_factors,
        'growth_factor': growth_factor,
        'daily_std': daily_std,
        'overall_std': all_historical['bookings'].std(),
        'years_trained': years,
        'total_days': len(all_historical)
    }

@st.cache_resource(show_spinner=False)
def analyze_prebookings(raw_data):
    """Analyze prebooking patterns"""
    if raw_data is not None and 'booking_created_date' in raw_data.columns:
        analyzer = PrebookingAnalyzer()
        analyzer.analyze_prebooking_patterns(raw_data)
        return analyzer
    return None

def predict_future_demand(model, prebooking_analyzer, start_date, 
                         prebooking_inputs=None, apply_holidays=True, 
                         confidence_level=95, use_prebooking=False):
    """Generate predictions with optional prebooking adjustments"""
    
    holiday_model = AssistedTravelHolidayPatterns() if apply_holidays else None
    dates = pd.date_range(start=start_date, periods=7, freq='D')
    predictions = []
    
    for date in dates:
        day_of_year = date.dayofyear
        day_of_week = date.dayofweek
        month = date.month
        
        # Base prediction
        base_value = model['seasonal_baseline'].get(day_of_year, 
                                                   np.mean(list(model['seasonal_baseline'].values())))
        dow_factor = model['dow_factors'].get(day_of_week, 1.0)
        month_factor = model['monthly_factors'].get(month, 1.0)
        
        holiday_factor = 1.0
        holiday_name = None
        if holiday_model:
            holiday_factor, holiday_name = holiday_model.get_holiday_factor(date)
        
        base_prediction = base_value * dow_factor * month_factor * model['growth_factor'] * holiday_factor
        
        # Confidence intervals
        std_dev = model['daily_std'].get(day_of_year, model['overall_std'])
        z_score = stats.norm.ppf((1 + confidence_level/100) / 2)
        margin = z_score * std_dev * 0.5
        
        lower_bound = max(0, base_prediction - margin)
        upper_bound = base_prediction + margin
        
        # Prebooking adjustment
        prebooking_adjustment = 1.0
        prebooking_flag = None
        prebooking_analysis = None
        final_prediction = base_prediction
        
        if use_prebooking and prebooking_analyzer and prebooking_inputs:
            current_prebookings = prebooking_inputs.get(date.date(), 0)
            days_until = (date.date() - datetime.now().date()).days
            
            if current_prebookings > 0 and days_until > 0 and days_until <= 7:
                adjustment_ratio, explanation, analysis = prebooking_analyzer.calculate_prebooking_adjustment(
                    day_of_week, current_prebookings, base_prediction, days_until
                )
                
                prebooking_analysis = analysis
                predicted_from_prebooking = analysis['predicted_final_demand'] if analysis else base_prediction
                
                if adjustment_ratio < 1.0:
                    prebooking_flag = f"⚠️ Lower demand signal: {explanation}"
                    final_prediction = base_prediction
                else:
                    if predicted_from_prebooking <= upper_bound:
                        final_prediction = predicted_from_prebooking
                        prebooking_flag = f"✅ Updated based on prebookings: {explanation}"
                        prebooking_adjustment = adjustment_ratio
                    else:
                        final_prediction = base_prediction
                        prebooking_flag = f"🔴 High prebooking signal (exceeds confidence): {explanation}"
        
        predictions.append({
            'date': date,
            'day_name': date.strftime('%A'),
            'base_prediction': round(base_prediction),
            'final_prediction': round(final_prediction),
            'lower_bound': round(lower_bound),
            'upper_bound': round(upper_bound),
            'base_value': round(base_value),
            'dow_factor': round(dow_factor, 3),
            'month_factor': round(month_factor, 3),
            'holiday_factor': round(holiday_factor, 3),
            'holiday_name': holiday_name,
            'growth_factor': round(model['growth_factor'], 3),
            'prebooking_adjustment': round(prebooking_adjustment, 3),
            'prebooking_flag': prebooking_flag,
            'prebooking_analysis': prebooking_analysis
        })
    
    return pd.DataFrame(predictions)

def categorize_demand(value, low_threshold, high_threshold):
    """Categorize demand level"""
    if value < low_threshold:
        return "Low", "🟢"
    elif value < high_threshold:
        return "Medium", "🟡"
    else:
        return "High", "🔴"

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.processed_years = None
    st.session_state.raw_data = None
    st.session_state.has_prebooking_data = False
    st.session_state.model = None
    st.session_state.prebooking_analyzer = None

# Load data on startup
if not st.session_state.data_loaded:
    processed_years, raw_data, has_prebooking_data = load_and_process_data()
    
    if processed_years and len(processed_years) >= 2:
        st.session_state.processed_years = processed_years
        st.session_state.raw_data = raw_data
        st.session_state.has_prebooking_data = has_prebooking_data
        st.session_state.data_loaded = True
        
        # Train model
        st.session_state.model = train_prediction_model(processed_years)
        
        # Analyze prebookings if available
        if raw_data is not None and has_prebooking_data:
            st.session_state.prebooking_analyzer = analyze_prebookings(raw_data)
        
        st.success(f"✅ Loaded {len(processed_years)} years of historical data")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("📅 Prediction Period")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() + timedelta(days=1),
            min_value=datetime.now().date(),
            help="First day of 7-day forecast"
        )
    
    with col2:
        end_date = start_date + timedelta(days=6)
        st.date_input(
            "End Date",
            value=end_date,
            disabled=True,
            help="Automatically set to 7 days"
        )
    
    st.info(f"📊 Forecasting {start_date.strftime('%a %d %b')} to {end_date.strftime('%a %d %b %Y')}")
    
    # Prebooking inputs
    days_until_start = (start_date - datetime.now().date()).days
    enable_prebooking = days_until_start >= 0 and days_until_start <= 7
    
    st.header("📋 Current Prebookings")
    
    prebooking_inputs = {}
    use_prebooking = False
    
    if st.session_state.data_loaded and enable_prebooking and st.session_state.has_prebooking_data:
        st.markdown("*Enter current booking numbers*")
        
        for i in range(7):
            date = start_date + timedelta(days=i)
            days_until = (date - datetime.now().date()).days
            
            if days_until > 0:
                prebooking_inputs[date] = st.number_input(
                    f"{date.strftime('%a %d/%m')} ({days_until}d away)",
                    min_value=0,
                    value=0,
                    step=1,
                    help=f"Bookings already made for {date.strftime('%A')}"
                )
        
        use_prebooking = st.checkbox(
            "Apply Prebooking Analysis",
            value=True,
            help="Adjust predictions based on current booking levels"
        )
    elif not enable_prebooking:
        st.info("Available for predictions within 7 days")
    elif not st.session_state.has_prebooking_data:
        st.warning("No booking date data available")
    
    st.header("⚙️ Settings")
    
    confidence_level = st.slider(
        "Confidence Interval (%)",
        80, 99, 95,
        help="Width of prediction intervals"
    )
    
    apply_holidays = st.checkbox(
        "Apply UK Holiday Patterns",
        value=True,
        help="Adjust for bank holiday travel"
    )
    
    show_components = st.checkbox(
        "Show Prediction Components",
        value=True,
        help="Display calculation breakdown"
    )
    
    st.header("🎯 Operational Thresholds")
    
    low_threshold = st.number_input(
        "Low Demand Threshold",
        min_value=0,
        value=150,
        help="Below = low staffing"
    )
    
    high_threshold = st.number_input(
        "High Demand Threshold", 
        min_value=0,
        value=300,
        help="Above = full staffing"
    )

# ============================================================================
# MAIN APPLICATION
# ============================================================================

if st.session_state.data_loaded and st.session_state.model:
    # Generate predictions
    with st.spinner("Generating predictions..."):
        predictions_df = predict_future_demand(
            st.session_state.model, 
            st.session_state.prebooking_analyzer,
            start_date, 
            prebooking_inputs=prebooking_inputs if use_prebooking else None,
            apply_holidays=apply_holidays,
            confidence_level=confidence_level,
            use_prebooking=use_prebooking
        )
    
    # Display results
    st.header("📊 7-Day Demand Forecast")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_predicted = predictions_df['final_prediction'].sum()
    avg_daily = predictions_df['final_prediction'].mean()
    peak_day = predictions_df.loc[predictions_df['final_prediction'].idxmax()]
    low_day = predictions_df.loc[predictions_df['final_prediction'].idxmin()]
    
    with col1:
        st.metric("Total Week Demand", f"{total_predicted:,.0f}")
    with col2:
        st.metric("Average Daily", f"{avg_daily:.0f}")
    with col3:
        st.metric("Peak Day", f"{peak_day['day_name'][:3]} ({peak_day['final_prediction']:,.0f})")
    with col4:
        st.metric("Lowest Day", f"{low_day['day_name'][:3]} ({low_day['final_prediction']:,.0f})")
    
    # Chart
    fig = go.Figure()
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=predictions_df['date'],
        y=predictions_df['upper_bound'],
        fill=None,
        mode='lines',
        line_color='rgba(0,100,80,0)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=predictions_df['date'],
        y=predictions_df['lower_bound'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,100,80,0)',
        name=f'{confidence_level}% Confidence Interval',
        fillcolor='rgba(68, 185, 255, 0.2)'
    ))
    
    # Base prediction line (if using prebooking)
    if use_prebooking and any(predictions_df['base_prediction'] != predictions_df['final_prediction']):
        fig.add_trace(go.Scatter(
            x=predictions_df['date'],
            y=predictions_df['base_prediction'],
            mode='lines+markers',
            name='Base Prediction',
            line=dict(color='gray', width=2, dash='dot'),
            marker=dict(size=8)
        ))
    
    # Final prediction line
    fig.add_trace(go.Scatter(
        x=predictions_df['date'],
        y=predictions_df['final_prediction'],
        mode='lines+markers',
        name='Final Prediction',
        line=dict(color='rgb(31, 119, 180)', width=3),
        marker=dict(size=10)
    ))
    
    # Threshold lines
    fig.add_hline(y=low_threshold, line_dash="dash", line_color="green", 
                 annotation_text="Low Threshold")
    fig.add_hline(y=high_threshold, line_dash="dash", line_color="red",
                 annotation_text="High Threshold")
    
    # Holiday markers
    holiday_dates = predictions_df[predictions_df['holiday_name'].notna()]
    if not holiday_dates.empty:
        fig.add_trace(go.Scatter(
            x=holiday_dates['date'],
            y=holiday_dates['final_prediction'],
            mode='markers',
            name='Holiday Impact',
            marker=dict(size=15, symbol='star', color='gold'),
            text=holiday_dates['holiday_name'],
            hovertemplate='%{text}<br>Demand: %{y}<extra></extra>'
        ))
    
    fig.update_layout(
        title="7-Day Demand Forecast",
        xaxis_title="Date",
        yaxis_title="Predicted Bookings",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Prebooking analysis results
    if use_prebooking and any(predictions_df['prebooking_flag'].notna()):
        st.header("📋 Prebooking Analysis")
        
        for _, row in predictions_df.iterrows():
            if row['prebooking_flag']:
                date_str = row['date'].strftime('%A %d %b')
                st.write(f"**{date_str}**: {row['prebooking_flag']}")
                
                if row['prebooking_analysis']:
                    analysis = row['prebooking_analysis']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Prebookings", 
                                 f"{analysis['actual_prebookings']:.0f}")
                    with col2:
                        st.metric("Expected Prebookings", 
                                 f"{analysis['expected_prebookings']:.0f}")
                    with col3:
                        st.metric("Implied Final Demand", 
                                 f"{analysis['predicted_final_demand']:.0f}")
    
    # Detailed table
    st.header("📋 Detailed Daily Predictions")
    
    display_df = predictions_df.copy()
    display_df['Demand Level'] = display_df['final_prediction'].apply(
        lambda x: categorize_demand(x, low_threshold, high_threshold)[1] + " " + 
                 categorize_demand(x, low_threshold, high_threshold)[0]
    )
    
    display_df['Date'] = display_df['date'].dt.strftime('%a %d %b')
    display_df['Prediction'] = display_df['final_prediction'].apply(lambda x: f"{x:,.0f}")
    display_df['Range'] = display_df.apply(
        lambda x: f"{x['lower_bound']:,.0f} - {x['upper_bound']:,.0f}", axis=1
    )
    display_df['Holiday'] = display_df['holiday_name'].fillna('-')
    
    display_columns = ['Date', 'Prediction', 'Range', 'Demand Level', 'Holiday']
    if use_prebooking:
        display_df['Prebooking Adj'] = display_df.apply(
            lambda x: f"{x['prebooking_adjustment']:.2f}x" if x['prebooking_adjustment'] != 1.0 else "-", 
            axis=1
        )
        display_columns.append('Prebooking Adj')
    
    st.dataframe(
        display_df[display_columns],
        use_container_width=True,
        hide_index=True
    )
    
    # Component breakdown
    if show_components:
        st.header("🔧 Prediction Components")
        
        components_df = predictions_df[[
            'date', 'base_value', 'dow_factor', 'month_factor', 
            'holiday_factor', 'growth_factor', 'final_prediction'
        ]].copy()
        
        if use_prebooking:
            components_df['prebooking_adjustment'] = predictions_df['prebooking_adjustment']
        
        components_df['Date'] = components_df['date'].dt.strftime('%a %d %b')
        components_df = components_df.drop('date', axis=1)
        
        column_names = {
            'Date': 'Date',
            'base_value': 'Seasonal Base', 
            'dow_factor': 'Day of Week',
            'month_factor': 'Monthly',
            'holiday_factor': 'Holiday',
            'growth_factor': 'Growth',
            'final_prediction': 'Final Prediction'
        }
        
        if use_prebooking:
            column_names['prebooking_adjustment'] = 'Prebooking'
        
        components_df.rename(columns={v: k for k, v in column_names.items()}, inplace=True)
        components_df.columns = list(column_names.values())
        
        st.dataframe(
            components_df,
            use_container_width=True,
            hide_index=True
        )
    
    # Operational recommendations
    st.header("🎯 Operational Recommendations")
    
    high_days = predictions_df[predictions_df['final_prediction'] > high_threshold]
    low_days = predictions_df[predictions_df['final_prediction'] < low_threshold]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚠️ High Demand Days")
        if not high_days.empty:
            for _, day in high_days.iterrows():
                st.warning(f"**{day['date'].strftime('%A %d %b')}**: {day['final_prediction']:,.0f} bookings expected")
            st.info("💡 Ensure full staffing and additional resources")
        else:
            st.success("No days exceed high threshold")
    
    with col2:
        st.subheader("✅ Low Demand Days")
        if not low_days.empty:
            for _, day in low_days.iterrows():
                st.info(f"**{day['date'].strftime('%A %d %b')}**: {day['final_prediction']:,.0f} bookings expected")
            st.success("💡 Opportunity for training or maintenance")
        else:
            st.info("No days below low threshold")
    
    # Export
    st.header("📥 Export Predictions")
    
    export_df = predictions_df[[
        'date', 'final_prediction', 'lower_bound', 
        'upper_bound', 'holiday_name', 'prebooking_flag'
    ]].copy()
    export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')
    export_df.columns = ['Date', 'Prediction', 'Lower_Bound', 'Upper_Bound', 'Holiday', 'Prebooking_Note']
    
    csv = export_df.to_csv(index=False)
    
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name=f"forecast_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload data files to begin prediction analysis.")
