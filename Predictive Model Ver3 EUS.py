import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import os
import requests
import io
import gzip
import pickle
import base64
import re

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
    st.warning("holidays package not installed. Install with: pip install holidays")

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
# DATA LOADING CONFIGURATION
# ============================================================================

# Option 1: Microsoft OneDrive Share Links
# Replace these with your actual OneDrive share links
ONEDRIVE_FILES = {
    "2023 Database.csv": "YOUR_ONEDRIVE_SHARE_LINK_HERE",  # Replace with actual share link
    "2024 Database.csv": "YOUR_ONEDRIVE_SHARE_LINK_HERE",  # Replace with actual share link
    "2025 Database.csv": "YOUR_ONEDRIVE_SHARE_LINK_HERE"   # Replace with actual share link
}

# Option 2: Direct URL downloads (Dropbox, GitHub Releases, etc.)
DIRECT_DOWNLOAD_URLS = {
    # "2023 Database.csv": "https://www.dropbox.com/s/xxxxx/2023_Database.csv?dl=1",
    # "2024 Database.csv": "https://github.com/yourusername/yourrepo/releases/download/v1.0/2024_Database.csv"
}

# Option 3: Compressed pickle files in repository
COMPRESSED_DATA_PATH = "compressed_data"

# Test mode - set to True to test OneDrive links
TEST_MODE = False

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def get_onedrive_direct_link(share_url):
    """Convert OneDrive share link to direct download link"""
    import re
    import base64
    
    # For OneDrive Personal links (1drv.ms)
    if "1drv.ms" in share_url:
        # These are shortened links, we need to get the actual URL
        try:
            # The most reliable method for OneDrive personal
            # Replace 'embed' with 'download' if present
            if "/embed" in share_url:
                return share_url.replace("/embed", "/download")
            
            # For regular share links, we need to transform them
            # OneDrive personal direct download format
            return share_url.replace("/redir", "/download").replace("/view", "/download") + "&download=1"
            
        except Exception as e:
            st.error(f"Error converting OneDrive personal link: {str(e)}")
            return None
    
    # For OneDrive Business/SharePoint links
    elif "sharepoint.com" in share_url:
        try:
            # Extract the necessary parts from the SharePoint URL
            if "/:u:/" in share_url or "/:x:/" in share_url or "/:t:/" in share_url:
                # This is already in a good format, just add download
                return share_url + "&download=1"
            else:
                # Convert regular sharepoint links
                # Replace /personal/ with /:f:/g/personal/ for files
                modified_url = share_url.replace("/personal/", "/:f:/g/personal/")
                # Add download parameter
                if "?" in modified_url:
                    return modified_url + "&download=1"
                else:
                    return modified_url + "?download=1"
                    
        except Exception as e:
            st.error(f"Error converting SharePoint link: {str(e)}")
            return None
    
    # For OneDrive embed links
    elif "onedrive.live.com" in share_url:
        try:
            if "embed" in share_url:
                # Extract the important parts and create download link
                cid_match = re.search(r'cid=([A-F0-9]+)', share_url, re.IGNORECASE)
                resid_match = re.search(r'resid=([A-F0-9%]+)', share_url, re.IGNORECASE)
                
                if cid_match and resid_match:
                    cid = cid_match.group(1)
                    resid = resid_match.group(1)
                    # Create direct download link
                    return f"https://onedrive.live.com/download?cid={cid}&resid={resid}"
                else:
                    # Fallback: try replacing embed with download
                    return share_url.replace("embed", "download")
            else:
                # Add download parameter
                if "?" in share_url:
                    return share_url + "&download=1"
                else:
                    return share_url + "?download=1"
                    
        except Exception as e:
            st.error(f"Error converting OneDrive live link: {str(e)}")
            return None
    
    else:
        # Unknown format, try adding download parameter
        if "?" in share_url:
            return share_url + "&download=1"
        else:
            return share_url + "?download=1"

def download_from_onedrive(share_link):
    """Download file from OneDrive using share link"""
    try:
        # Get direct download link
        direct_link = get_onedrive_direct_link(share_link)
        
        if not direct_link:
            st.error("Could not convert OneDrive link to download format")
            return None
        
        # Headers to mimic browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Create session for better cookie handling
        session = requests.Session()
        session.headers.update(headers)
        
        # First, try the direct link
        response = session.get(direct_link, allow_redirects=True, timeout=60)
        
        # Check if we got actual CSV content
        content_type = response.headers.get('content-type', '').lower()
        
        if response.status_code == 200:
            # Check if it's actually CSV data
            try:
                # Try to decode first 1000 bytes as text
                sample = response.content[:1000].decode('utf-8', errors='ignore')
                
                # Check for CSV indicators
                if any(indicator in sample for indicator in [',', '\t', 'station_code', 'booking']):
                    return response.content
                elif '<html' in sample.lower():
                    # We got an HTML page instead of the file
                    st.error("""
                    ❌ OneDrive returned a web page instead of the CSV file.
                    
                    Please try this method to get the correct link:
                    1. Open the file in OneDrive
                    2. Click the "Download" button (not share)
                    3. While download starts, right-click and copy the download link
                    4. Use that direct download link instead
                    """)
                    return None
            except:
                # If we can't decode as text, might still be valid binary CSV
                pass
        
        # If first attempt failed, try alternative methods
        if response.status_code != 200 or 'html' in content_type:
            # Try with different parameters
            alternative_urls = [
                share_link + "&download=1",
                share_link.replace("view?", "download?"),
                share_link.replace("/view", "/download"),
                share_link.replace("?e=", "/download?e=")
            ]
            
            for alt_url in alternative_urls:
                try:
                    response = session.get(alt_url, allow_redirects=True, timeout=60)
                    if response.status_code == 200:
                        sample = response.content[:1000].decode('utf-8', errors='ignore')
                        if not '<html' in sample.lower():
                            return response.content
                except:
                    continue
        
        st.error(f"""
        ❌ Failed to download from OneDrive (Status: {response.status_code})
        
        Common issues:
        - File might be in a folder (need direct file link)
        - Permissions might be restricted
        - Link might have expired
        
        Try getting a fresh share link with "Anyone with link can view" permission.
        """)
        
        return None
            
    except requests.exceptions.Timeout:
        st.error("Download timed out. The file might be too large or the connection is slow.")
        return None
    except Exception as e:
        st.error(f"Error downloading from OneDrive: {str(e)}")
        return None

def download_from_url(url):
    """Download file from direct URL"""
    try:
        response = requests.get(url, timeout=300)  # 5 minute timeout for large files
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"Error downloading file: HTTP {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error downloading from URL: {str(e)}")
        return None

@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def load_cloud_data():
    """Load data from cloud storage (OneDrive, Dropbox, etc.)"""
    data_files = []
    
    # Try OneDrive files first
    if ONEDRIVE_FILES and any(link != "YOUR_ONEDRIVE_SHARE_LINK_HERE" for link in ONEDRIVE_FILES.values()):
        for filename, share_link in ONEDRIVE_FILES.items():
            if share_link != "YOUR_ONEDRIVE_SHARE_LINK_HERE":
                with st.spinner(f"Downloading {filename} from OneDrive..."):
                    content = download_from_onedrive(share_link)
                    if content:
                        # Verify it's CSV content
                        try:
                            # Try to parse first few lines to verify it's CSV
                            test_df = pd.read_csv(io.BytesIO(content), nrows=5)
                            st.success(f"✓ Successfully downloaded {filename}")
                            data_files.append({
                                'name': filename,
                                'data': content
                            })
                        except Exception as e:
                            st.error(f"""
                            ❌ {filename} downloaded but is not a valid CSV file.
                            
                            **Common causes:**
                            - File is Excel format (.xlsx) instead of CSV
                            - File is corrupted or empty
                            - Wrong file was shared
                            
                            Please verify the file is in CSV format and try again.
                            """)
                    else:
                        st.error(f"""
                        ❌ Failed to download {filename}
                        
                        **Try this:**
                        1. Open the file in OneDrive
                        2. Click "Download" button
                        3. Right-click the download → "Copy download link"
                        4. Use that link instead
                        
                        Or use the "Test OneDrive Link" feature in the sidebar.
                        """)
    
    # Try direct URLs
    elif DIRECT_DOWNLOAD_URLS:
        for filename, url in DIRECT_DOWNLOAD_URLS.items():
            st.info(f"Downloading {filename}...")
            content = download_from_url(url)
            if content:
                data_files.append({
                    'name': filename,
                    'data': content
                })
    
    return data_files

@st.cache_data(show_spinner=False)
def load_compressed_data():
    """Load pre-processed compressed data from repository"""
    try:
        # Check if compressed data directory exists
        if os.path.exists(COMPRESSED_DATA_PATH):
            compressed_files = [f for f in os.listdir(COMPRESSED_DATA_PATH) if f.endswith('.pkl.gz')]
            
            if compressed_files:
                data = {}
                for file in compressed_files:
                    file_path = os.path.join(COMPRESSED_DATA_PATH, file)
                    with gzip.open(file_path, 'rb') as f:
                        year = int(file.split('_')[0])
                        data[year] = pickle.load(f)
                
                return data
    except Exception as e:
        st.error(f"Error loading compressed data: {str(e)}")
    
    return None

def save_compressed_data(processed_years, output_dir="compressed_data"):
    """Save processed data as compressed pickle files (for local preprocessing)"""
    os.makedirs(output_dir, exist_ok=True)
    
    for year, data in processed_years.items():
        output_file = os.path.join(output_dir, f"{year}_processed.pkl.gz")
        with gzip.open(output_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {output_file}")

# ============================================================================
# STREAMLIT SECRETS CONFIGURATION
# ============================================================================

def load_from_secrets():
    """Load configuration from Streamlit secrets"""
    data_files = []
    
    # Check if secrets are configured
    if hasattr(st, 'secrets') and 'data_sources' in st.secrets:
        sources = st.secrets['data_sources']
        
        # OneDrive files
        if 'onedrive' in sources:
            st.info("Loading files from OneDrive (configured in Secrets)...")
            for filename, share_link in sources['onedrive'].items():
                with st.spinner(f"Downloading {filename} from OneDrive..."):
                    content = download_from_onedrive(share_link)
                    if content:
                        # Verify it's CSV content
                        try:
                            # Try to parse first few lines to verify it's CSV
                            test_df = pd.read_csv(io.BytesIO(content), nrows=5)
                            st.success(f"✓ Successfully downloaded {filename}")
                            data_files.append({
                                'name': filename,
                                'data': content
                            })
                        except Exception as e:
                            st.error(f"""
                            ❌ {filename} is not a valid CSV file.
                            
                            Please check:
                            - File is in CSV format (not .xlsx)
                            - File is not corrupted
                            - Correct file was shared
                            """)
                    else:
                        st.error(f"""
                        ❌ Failed to download {filename}
                        
                        Check your OneDrive link in Streamlit Secrets.
                        Use the "Test OneDrive Link" feature to verify.
                        """)
        
        # Direct URLs
        elif 'urls' in sources:
            for filename, url in sources['urls'].items():
                st.info(f"Downloading {filename}...")
                content = download_from_url(url)
                if content:
                    data_files.append({
                        'name': filename,
                        'data': content
                    })
    
    return data_files

# ============================================================================
# DATA PROCESSING (keeping all original functions)
# ============================================================================

@st.cache_data(show_spinner=False)
def process_data_files(data_files):
    """Process data files (either default or uploaded)"""
    yearly_data = {}
    has_prebooking_data = False
    
    for file_info in data_files:
        try:
            # Extract year from filename
            file_name = file_info['name'] if isinstance(file_info, dict) else file_info.name
            year = int(file_name.split()[0])
            
            # Read CSV data
            if isinstance(file_info, dict):
                # Default data (bytes)
                df = pd.read_csv(io.BytesIO(file_info['data']), low_memory=False)
            else:
                # Uploaded file
                df = pd.read_csv(file_info, low_memory=False)
            
            # Filter for Euston
            df_euston = df[df['station_code'] == "EUS"].copy()
            
            # Check if prebooking data is available
            if 'booking_created_date' in df_euston.columns:
                has_prebooking_data = True
            
            if len(df_euston) > 0:
                yearly_data[year] = df_euston
                
        except Exception as e:
            st.error(f"Error processing {file_name}: {str(e)}")
            continue
    
    if not yearly_data:
        return None, None, None
    
    # Process each year
    processed_years = {}
    
    for year, df in yearly_data.items():
        # Convert dates
        df['scheduled_departure_date'] = pd.to_datetime(
            df['scheduled_departure_date'], 
            dayfirst=True,
            errors='coerce'
        )
        
        # Convert booking_created_date if present
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
        
        # Handle partial years
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
    
    # Combine all raw data if prebooking analysis needed
    all_raw_data = None
    if has_prebooking_data:
        all_raw_data = pd.concat([yearly_data[year] for year in yearly_data.keys()])
    
    return processed_years, all_raw_data, has_prebooking_data

# [Include all the original classes and functions here - PrebookingAnalyzer, AssistedTravelHolidayPatterns, etc.]
# [I'm not repeating them to save space, but they should all be included]

# ============================================================================
# PREBOOKING ANALYZER CLASS
# ============================================================================

class PrebookingAnalyzer:
    """Analyses relationship between prebookings and final demand"""
    
    def __init__(self):
        self.prebooking_models = {}  # Store models by day of week
        self.prebooking_stats = {}   # Store statistics
        self.day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    def analyze_prebooking_patterns(self, historical_data):
        """Analyse prebooking patterns from historical data"""
        
        # For each booking, calculate days in advance
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
            
            # Skip if too few bookings
            if len(date_bookings) < 10:
                continue
            
            # Calculate cumulative bookings at each day before departure
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
            
            # Calculate statistics for each days_before cutoff
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
        
        # Expected prebookings based on base prediction and historical patterns
        expected_percentage = stats['mean_percentage']
        expected_prebookings = base_prediction * expected_percentage
        
        # Calculate predicted final demand based on current prebookings
        if expected_percentage > 0:
            predicted_final_demand = current_prebookings / expected_percentage
        else:
            predicted_final_demand = base_prediction
        
        # Calculate how much above/below the base prediction
        if base_prediction > 0:
            adjustment_ratio = predicted_final_demand / base_prediction
        else:
            adjustment_ratio = 1.0
        
        # Prepare detailed analysis
        analysis = {
            'expected_prebookings': expected_prebookings,
            'actual_prebookings': current_prebookings,
            'expected_percentage': expected_percentage,
            'predicted_final_demand': predicted_final_demand,
            'base_prediction': base_prediction,
            'adjustment_ratio': adjustment_ratio
        }
        
        # Create explanation
        if adjustment_ratio > 1.0:
            explanation = f"Prebookings {current_prebookings:.0f} vs {expected_prebookings:.0f} expected → Predicting {predicted_final_demand:.0f} total"
        elif adjustment_ratio < 1.0:
            explanation = f"Prebookings {current_prebookings:.0f} vs {expected_prebookings:.0f} expected → Lower demand signal"
        else:
            explanation = "Prebookings match expectations"
        
        return adjustment_ratio, explanation, analysis

# ============================================================================
# HOLIDAY PATTERN CLASS (from original)
# ============================================================================

class AssistedTravelHolidayPatterns:
    """Handles UK bank holiday patterns specific to assisted travel customers"""
    
    def __init__(self):
        # Get base holidays from package if available
        current_year = datetime.now().year
        if HOLIDAYS_AVAILABLE:
            self.uk_holidays = holidays.UK(years=range(current_year-2, current_year+2))
        else:
            # Manual fallback for key holidays
            self.uk_holidays = self._get_manual_holidays()
        
        # Define patterns specific to assisted travel
        self.pattern_templates = {
            'easter': {
                'good_friday': {
                    -2: 1.3,   # Thu before - outbound travel
                    -1: 1.35,  # Day before (Thu) - peak outbound
                    0: 1.3,    # Good Friday - still high
                    1: 0.85,   # Saturday - low (middle of holiday)
                    2: 0.8     # Easter Sunday - lowest
                },
                'easter_monday': {
                    -2: 0.8,   # Saturday - low
                    -1: 0.85,  # Sunday - low
                    0: 1.4,    # Easter Monday - return surge
                    1: 1.35,   # Tuesday - continued returns
                    2: 1.2     # Wednesday - normalizing
                }
            },
            'standard_monday': {  # Most bank holidays are Mondays
                -3: 1.2,   # Friday before - early outbound
                -2: 1.25,  # Weekend before - moderate
                -1: 1.0,   # Sunday - lower
                0: 0.9,    # Monday holiday - lower
                1: 1.4,    # Tuesday - return surge
                2: 1.25    # Wednesday - still elevated
            },
            'christmas': {
                -3: 1.35,  # Pre-Christmas travel
                -2: 1.4,   # Peak pre-Christmas
                -1: 1.45,  # Christmas Eve travel
                0: 0.7,    # Christmas Day - minimal
                1: 0.8,    # Boxing Day - low
                2: 1.1,    # 27th - some returns
                3: 1.2     # 28th - more returns
            },
            'new_year': {
                -2: 1.2,   # 30th Dec - outbound
                -1: 1.25,  # NYE - moderate
                0: 0.8,    # New Year's Day - low
                1: 1.1,    # 2nd Jan - returns begin
                2: 1.3     # 3rd Jan - return surge
            }
        }
    
    def _get_manual_holidays(self):
        """Manual holiday definitions as fallback"""
        current_year = datetime.now().year
        holidays_dict = {}
        
        # Add holidays for current and next year
        for year in [current_year, current_year + 1]:
            holidays_dict.update({
                pd.Timestamp(f'{year}-01-01'): "New Year's Day",
                pd.Timestamp(f'{year}-12-25'): 'Christmas Day',
                pd.Timestamp(f'{year}-12-26'): 'Boxing Day'
            })
            
            # Add Easter (approximate - would need proper calculation)
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
        
        # Convert to pandas timestamp if needed
        if not isinstance(check_date, pd.Timestamp):
            check_date = pd.Timestamp(check_date)
        
        # Check proximity to holidays
        best_factor = 1.0
        holiday_name = None
        
        for holiday_date, name in self.uk_holidays.items():
            if not isinstance(holiday_date, pd.Timestamp):
                holiday_date = pd.Timestamp(holiday_date)
                
            days_diff = (check_date - holiday_date).days
            
            # Skip if too far from holiday
            if abs(days_diff) > 5:
                continue
                
            # Get appropriate pattern
            pattern = self.get_pattern_for_holiday(name, holiday_date)
            
            if days_diff in pattern:
                factor = pattern[days_diff]
                
                # Adjust for day of week preferences
                if check_date.weekday() in [5, 6]:  # Weekend
                    factor *= 0.9  # Assisted travel avoids weekends
                elif check_date.weekday() in [1, 2, 3]:  # Tue-Thu
                    factor *= 1.05  # Slight boost for preferred days
                
                # Take the maximum factor if multiple holidays affect this date
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
            # Default bank holiday pattern
            return {-2: 1.2, -1: 1.25, 0: 1.1, 1: 1.25, 2: 1.2}

# ============================================================================
# MODEL TRAINING AND PREDICTION FUNCTIONS
# ============================================================================

@st.cache_resource(show_spinner=False)
def train_cached_model(processed_years):
    """Train and cache the prediction model"""
    return train_prediction_model(processed_years)

@st.cache_resource(show_spinner=False)
def analyze_cached_prebookings(raw_data):
    """Analyze and cache prebooking patterns"""
    if raw_data is not None and 'booking_created_date' in raw_data.columns:
        analyzer = PrebookingAnalyzer()
        analyzer.analyze_prebooking_patterns(raw_data)
        return analyzer
    return None

def train_prediction_model(processed_years):
    """Train the base model on all historical data"""
    
    # Combine all historical data
    all_historical = pd.concat(processed_years.values())
    
    # Calculate growth trend
    yearly_totals = []
    for year, data in processed_years.items():
        total = data['bookings'].sum()
        yearly_totals.append((year, total))
    
    years = [x[0] for x in yearly_totals]
    totals = [x[1] for x in yearly_totals]
    
    # Calculate growth factor (adaptive method)
    if len(years) >= 2:
        # Year-over-year growth rates
        growth_rates = []
        for i in range(1, len(yearly_totals)):
            prev_total = yearly_totals[i-1][1]
            curr_total = yearly_totals[i][1]
            growth = (curr_total - prev_total) / prev_total if prev_total > 0 else 0
            growth_rates.append(growth)
        
        # Weight recent years more
        if len(growth_rates) > 1:
            weights = [0.7 ** (len(growth_rates) - i - 1) for i in range(len(growth_rates))]
            weights = [w / sum(weights) for w in weights]
            avg_growth = sum(g * w for g, w in zip(growth_rates, weights))
        else:
            avg_growth = growth_rates[0]
        
        # Project to current year
        years_ahead = datetime.now().year - years[-1]
        growth_factor = (1 + avg_growth) ** years_ahead
    else:
        growth_factor = 1.1
    
    # Calculate seasonal baseline (median approach)
    seasonal_baseline = all_historical.groupby('day_of_year')['bookings'].median().to_dict()
    
    # Fill missing days with interpolation
    all_days = range(1, 367)
    for day in all_days:
        if day not in seasonal_baseline:
            # Find nearest days
            nearby_days = [d for d in seasonal_baseline.keys() if abs(d - day) <= 7]
            if nearby_days:
                seasonal_baseline[day] = np.mean([seasonal_baseline[d] for d in nearby_days])
            else:
                seasonal_baseline[day] = all_historical['bookings'].median()
    
    # Calculate day-of-week factors
    dow_medians = all_historical.groupby('day_of_week')['bookings'].median()
    overall_median = all_historical['bookings'].median()
    dow_factors = (dow_medians / overall_median).to_dict() if overall_median > 0 else {}
    
    # Calculate monthly factors
    monthly_medians = all_historical.groupby('month')['bookings'].median()
    monthly_factors = (monthly_medians / overall_median).to_dict() if overall_median > 0 else {}
    
    # Calculate prediction intervals based on historical variance
    daily_std = all_historical.groupby('day_of_year')['bookings'].std().to_dict()
    
    model = {
        'seasonal_baseline': seasonal_baseline,
        'dow_factors': dow_factors,
        'monthly_factors': monthly_factors,
        'growth_factor': growth_factor,
        'daily_std': daily_std,
        'overall_std': all_historical['bookings'].std(),
        'years_trained': years,
        'total_days': len(all_historical)
    }
    
    return model

def predict_future_demand_with_prebooking(model, prebooking_analyzer, start_date, 
                                         prebooking_inputs=None, apply_holidays=True, 
                                         confidence_level=95, use_prebooking=False):
    """Generate predictions with optional prebooking adjustments"""
    
    # Initialize holiday patterns
    holiday_model = AssistedTravelHolidayPatterns() if apply_holidays else None
    
    # Generate date range
    dates = pd.date_range(start=start_date, periods=7, freq='D')
    
    predictions = []
    
    for date in dates:
        # Get base components
        day_of_year = date.dayofyear
        day_of_week = date.dayofweek
        month = date.month
        
        # Seasonal baseline
        base_value = model['seasonal_baseline'].get(day_of_year, 
                                                   np.mean(list(model['seasonal_baseline'].values())))
        
        # Apply factors
        dow_factor = model['dow_factors'].get(day_of_week, 1.0)
        month_factor = model['monthly_factors'].get(month, 1.0)
        
        # Holiday factor
        holiday_factor = 1.0
        holiday_name = None
        if holiday_model:
            holiday_factor, holiday_name = holiday_model.get_holiday_factor(date)
        
        # Combine all factors for base prediction
        base_prediction = base_value * dow_factor * month_factor * model['growth_factor'] * holiday_factor
        
        # Calculate confidence intervals
        std_dev = model['daily_std'].get(day_of_year, model['overall_std'])
        z_score = stats.norm.ppf((1 + confidence_level/100) / 2)
        margin = z_score * std_dev * 0.5  # Reduce margin for more realistic intervals
        
        lower_bound = max(0, base_prediction - margin)
        upper_bound = base_prediction + margin
        
        # Initialize prebooking analysis
        prebooking_adjustment = 1.0
        prebooking_flag = None
        prebooking_analysis = None
        final_prediction = base_prediction
        
        # Apply prebooking analysis if enabled and available
        if use_prebooking and prebooking_analyzer and prebooking_inputs:
            current_prebookings = prebooking_inputs.get(date.date(), 0)
            days_until = (date.date() - datetime.now().date()).days
            
            if current_prebookings > 0 and days_until > 0 and days_until <= 7:
                adjustment_ratio, explanation, analysis = prebooking_analyzer.calculate_prebooking_adjustment(
                    day_of_week, current_prebookings, base_prediction, days_until
                )
                
                prebooking_analysis = analysis
                predicted_from_prebooking = analysis['predicted_final_demand'] if analysis else base_prediction
                
                # Apply business rules
                if adjustment_ratio < 1.0:
                    # Lower than expected - just flag it
                    prebooking_flag = f"⚠️ Lower demand signal: {explanation}"
                    final_prediction = base_prediction  # Keep original
                else:
                    # Higher than expected
                    if predicted_from_prebooking <= upper_bound:
                        # Within confidence interval - update prediction
                        final_prediction = predicted_from_prebooking
                        prebooking_flag = f"✅ Updated based on prebookings: {explanation}"
                        prebooking_adjustment = adjustment_ratio
                    else:
                        # Outside confidence interval - keep original but flag
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
    """Categorize demand level for operational planning"""
    if value < low_threshold:
        return "Low", "🟢"
    elif value < high_threshold:
        return "Medium", "🟡"
    else:
        return "High", "🔴"

# ============================================================================
# LOAD DATA ON STARTUP
# ============================================================================

# Initialize session state for data
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.processed_years = None
    st.session_state.raw_data = None
    st.session_state.has_prebooking_data = False
    st.session_state.model = None
    st.session_state.prebooking_analyzer = None
    st.session_state.use_uploaded_data = False
    st.session_state.data_source = None

# Load data on startup
if not st.session_state.data_loaded and not st.session_state.use_uploaded_data:
    with st.spinner("Loading historical data..."):
        
        # Try different data sources in order
        data_loaded = False
        error_messages = []
        
        # 1. Try Streamlit secrets first
        if not data_loaded:
            try:
                data_files = load_from_secrets()
                if data_files:
                    st.session_state.data_source = "Streamlit Secrets"
                    data_loaded = True
            except Exception as e:
                error_messages.append(f"Secrets: {str(e)}")
        
        # 2. Try compressed data
        if not data_loaded:
            compressed_data = load_compressed_data()
            if compressed_data:
                st.session_state.processed_years = compressed_data
                st.session_state.data_loaded = True
                st.session_state.data_source = "Compressed Files"
                st.session_state.has_prebooking_data = False  # Compressed data doesn't include raw data
                
                # Train model
                st.session_state.model = train_cached_model(compressed_data)
                st.success(f"✅ Loaded pre-processed data from {len(compressed_data)} years")
                data_loaded = True
        
        # 3. Try cloud storage
        if not data_loaded:
            try:
                data_files = load_cloud_data()
                if data_files:
                    st.session_state.data_source = "Cloud Storage"
                    data_loaded = True
            except Exception as e:
                error_messages.append(f"Cloud Storage: {str(e)}")
        
        # Process data files if loaded
        if data_loaded and not st.session_state.data_loaded and 'data_files' in locals() and data_files:
            try:
                processed_years, raw_data, has_prebooking_data = process_data_files(data_files)
                
                if processed_years and len(processed_years) >= 2:
                    st.session_state.processed_years = processed_years
                    st.session_state.raw_data = raw_data
                    st.session_state.has_prebooking_data = has_prebooking_data
                    st.session_state.data_loaded = True
                    
                    # Train model
                    st.session_state.model = train_cached_model(processed_years)
                    
                    # Analyze prebookings if available
                    if raw_data is not None and has_prebooking_data:
                        st.session_state.prebooking_analyzer = analyze_cached_prebookings(raw_data)
                    
                    st.success(f"✅ Loaded {len(processed_years)} years of data from {st.session_state.data_source}")
                else:
                    st.warning("Need at least 2 years of historical data")
                    if error_messages:
                        with st.expander("🔍 Error Details"):
                            for msg in error_messages:
                                st.text(msg)
            except Exception as e:
                st.error(f"Error processing data files: {str(e)}")
                error_messages.append(f"Processing: {str(e)}")

# ============================================================================
# SIDEBAR (rest of the code remains the same)
# ============================================================================

with st.sidebar:
    st.header("📁 Data Source")
    
    if st.session_state.data_source:
        st.info(f"Using: {st.session_state.data_source}")
    
    # Option to use uploaded data instead
    use_custom_data = st.checkbox(
        "Upload Custom Data", 
        value=st.session_state.use_uploaded_data,
        help="Upload your own data files instead of using pre-loaded data"
    )
    
    if use_custom_data:
        st.session_state.use_uploaded_data = True
        uploaded_files = st.file_uploader(
            "Upload Historical Database CSV Files",
            accept_multiple_files=True,
            type="csv",
            help="Upload 2+ years of historical data with booking_created_date column"
        )
        
        if uploaded_files and st.button("Process Uploaded Data"):
            with st.spinner("Processing uploaded data..."):
                processed_years, raw_data, has_prebooking_data = process_data_files(uploaded_files)
                
                if processed_years and len(processed_years) >= 2:
                    st.session_state.processed_years = processed_years
                    st.session_state.raw_data = raw_data
                    st.session_state.has_prebooking_data = has_prebooking_data
                    st.session_state.data_loaded = True
                    st.session_state.data_source = "Uploaded Files"
                    
                    # Clear cached models to retrain
                    st.cache_resource.clear()
                    
                    # Train model
                    st.session_state.model = train_cached_model(processed_years)
                    
                    # Analyze prebookings if available
                    if raw_data is not None and has_prebooking_data:
                        st.session_state.prebooking_analyzer = analyze_cached_prebookings(raw_data)
                    
                    st.success(f"✅ Processed {len(uploaded_files)} files successfully")
                else:
                    st.error("Need at least 2 years of historical data")
    # Option to test OneDrive links
    with st.expander("🧪 Test OneDrive Link"):
        st.markdown("Test if your OneDrive link works before adding to Secrets")
        
        test_link = st.text_input(
            "Paste OneDrive share link to test:",
            placeholder="https://1drv.ms/..."
        )
        
        if st.button("Test Link"):
            if test_link:
                with st.spinner("Testing OneDrive link..."):
                    try:
                        # Get direct link
                        direct_link = get_onedrive_direct_link(test_link)
                        st.info(f"Converted to: {direct_link[:50]}...")
                        
                        # Try to download
                        content = download_from_onedrive(test_link)
                        
                        if content:
                            # Check if it's CSV
                            try:
                                df_test = pd.read_csv(io.BytesIO(content), nrows=5)
                                st.success("✅ Link works! File appears to be valid CSV")
                                st.write("First few rows:")
                                st.dataframe(df_test)
                            except:
                                st.error("❌ File downloaded but doesn't appear to be valid CSV")
                        else:
                            st.error("❌ Could not download file from this link")
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("Please enter a link to test")
    
    # Add configuration help
    with st.expander("🔧 OneDrive Setup Guide"):
        st.markdown("""
        ### Getting the Right OneDrive Link
        
        **For OneDrive Personal:**
        1. Upload your CSV files to OneDrive
        2. Right-click the file → "Share"
        3. Click "Anyone with the link can view"
        4. Click "Copy link"
        5. The link should look like: `https://1drv.ms/...`
        
        **For OneDrive Business/SharePoint:**
        1. Upload CSV files to OneDrive for Business
        2. Select the file → Click "Share" button
        3. Under "Link settings":
           - Choose "Anyone with the link"
           - Uncheck "Allow editing"
           - Click "Apply"
        4. Click "Copy link"
        
        **Important Tips:**
        ✅ Share individual CSV files, not folders
        ✅ Make sure files are .csv format, not .xlsx
        ✅ Set permission to "Anyone with the link can view"
        ✅ Test the link in an incognito browser window
        
        **Alternative Method (Most Reliable):**
        1. Open the CSV file in OneDrive web interface
        2. Click "Download" button at the top
        3. When download starts, right-click → "Copy download link"
        4. Use this direct download link instead
        
        **Configure in Streamlit Secrets:**
        ```toml
        [data_sources.onedrive]
        "2023 Database.csv" = "paste_your_link_here"
        "2024 Database.csv" = "paste_your_link_here"
        "2025 Database.csv" = "paste_your_link_here"
        ```
        """)
    
    st.header("📅 Prediction Period")
    
    # Date selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() + timedelta(days=1),
            min_value=datetime.now().date(),
            help="Select the first day of your 7-day forecast"
        )
    
    with col2:
        # Automatically set end date to 6 days after start
        end_date = start_date + timedelta(days=6)
        st.date_input(
            "End Date",
            value=end_date,
            disabled=True,
            help="Automatically set to 7 days"
        )
    
    st.info(f"📊 Forecasting {start_date.strftime('%a %d %b')} to {end_date.strftime('%a %d %b %Y')}")
    
    # Check if forecast is within next 7 days
    days_until_start = (start_date - datetime.now().date()).days
    enable_prebooking = days_until_start >= 0 and days_until_start <= 7
    
    # Prebooking inputs section
    st.header("📋 Current Prebookings")
    
    if not st.session_state.data_loaded:
        st.info("Load data first to enable prebooking analysis")
    elif enable_prebooking:
        if st.session_state.has_prebooking_data:
            st.markdown("*Enter current prebooking numbers for enhanced predictions*")
            
            prebooking_inputs = {}
            for i in range(7):
                date = start_date + timedelta(days=i)
                days_until = (date - datetime.now().date()).days
                
                if days_until > 0:
                    prebooking_inputs[date] = st.number_input(
                        f"{date.strftime('%a %d/%m')} ({days_until}d away)",
                        min_value=0,
                        value=0,
                        step=1,
                        help=f"How many bookings already made for {date.strftime('%A')}?"
                    )
            
            use_prebooking = st.checkbox(
                "Apply Prebooking Analysis",
                value=True,
                help="Adjust predictions based on current booking levels"
            )
            
            st.info("""
            💡 **How it works:**
            - Enter the number of bookings already made for each day
            - The model compares this to historical patterns
            - Predictions adjust if bookings are unusually high/low
            """)
        else:
            st.warning("""
            ⚠️ Prebooking analysis unavailable
            
            The historical data doesn't include booking creation dates.
            To enable this feature, your CSV files need a 
            'booking_created_date' column.
            """)
            prebooking_inputs = {}
            use_prebooking = False
    else:
        st.info("Prebooking analysis available for predictions within next 7 days")
        prebooking_inputs = {}
        use_prebooking = False
    
    st.header("⚙️ Model Settings")
    
    confidence_level = st.slider(
        "Confidence Interval (%)",
        80, 99, 95,
        help="Confidence level for prediction intervals"
    )
    
    apply_holidays = st.checkbox(
        "Apply UK Holiday Patterns",
        value=True,
        help="Apply assisted travel holiday behavior"
    )
    
    show_components = st.checkbox(
        "Show Prediction Components",
        value=True,
        help="Break down how each prediction is calculated"
    )
    
    st.header("🎯 Operational Thresholds")
    
    low_threshold = st.number_input(
        "Low Demand Threshold",
        min_value=0,
        value=150,
        help="Below this = low staffing needed"
    )
    
    high_threshold = st.number_input(
        "High Demand Threshold", 
        min_value=0,
        value=300,
        help="Above this = full staffing needed"
    )

# ============================================================================
# MAIN APPLICATION (same as before)
# ============================================================================

def main():
    try:
        if st.session_state.data_loaded and st.session_state.model:
            # Generate predictions
            with st.spinner("Generating predictions..."):
                predictions_df = predict_future_demand_with_prebooking(
                    st.session_state.model, 
                    st.session_state.prebooking_analyzer,
                    start_date, 
                    prebooking_inputs=prebooking_inputs if use_prebooking else None,
                    apply_holidays=apply_holidays,
                    confidence_level=confidence_level,
                    use_prebooking=use_prebooking and st.session_state.prebooking_analyzer is not None
                )
            
            # Display predictions
            st.header("📊 7-Day Demand Forecast")
            
            # Show model info
            st.info(f"Model trained on {len(st.session_state.model['years_trained'])} years: {min(st.session_state.model['years_trained'])}-{max(st.session_state.model['years_trained'])}")
            
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
            
            # Visualization
            fig = go.Figure()
            
            # Add confidence interval
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
            
            # Base prediction line (if different from final)
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
            
            # Add threshold lines
            fig.add_hline(y=low_threshold, line_dash="dash", line_color="green", 
                         annotation_text="Low Threshold")
            fig.add_hline(y=high_threshold, line_dash="dash", line_color="red",
                         annotation_text="High Threshold")
            
            # Mark holidays
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
                title="7-Day Demand Forecast with Prebooking Intelligence",
                xaxis_title="Date",
                yaxis_title="Predicted Bookings",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prebooking Analysis Results
            if use_prebooking and any(predictions_df['prebooking_flag'].notna()):
                st.header("📋 Prebooking Analysis Results")
                
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
            
            # Detailed predictions table
            st.header("📋 Detailed Daily Predictions")
            
            # Prepare display dataframe
            display_df = predictions_df.copy()
            
            # Add demand category
            display_df['Demand Level'] = display_df['final_prediction'].apply(
                lambda x: categorize_demand(x, low_threshold, high_threshold)[1] + " " + 
                         categorize_demand(x, low_threshold, high_threshold)[0]
            )
            
            # Format columns
            display_df['Date'] = display_df['date'].dt.strftime('%a %d %b')
            display_df['Base Prediction'] = display_df['base_prediction'].apply(lambda x: f"{x:,.0f}")
            display_df['Final Prediction'] = display_df['final_prediction'].apply(lambda x: f"{x:,.0f}")
            display_df['Range'] = display_df.apply(
                lambda x: f"{x['lower_bound']:,.0f} - {x['upper_bound']:,.0f}", axis=1
            )
            display_df['Holiday'] = display_df['holiday_name'].fillna('-')
            display_df['Prebooking Adj'] = display_df.apply(
                lambda x: f"{x['prebooking_adjustment']:.2f}x" if x['prebooking_adjustment'] != 1.0 else "-", 
                axis=1
            )
            
            # Select columns to display
            display_columns = ['Date', 'Base Prediction', 'Final Prediction', 'Range', 
                              'Demand Level', 'Holiday', 'Prebooking Adj']
            
            st.dataframe(
                display_df[display_columns],
                use_container_width=True,
                hide_index=True
            )
            
            # Component breakdown
            if show_components:
                st.header("🔧 Prediction Components Breakdown")
                
                components_df = predictions_df[[
                    'date', 'base_value', 'dow_factor', 'month_factor', 
                    'holiday_factor', 'growth_factor', 'prebooking_adjustment', 'final_prediction'
                ]].copy()
                
                components_df['Date'] = components_df['date'].dt.strftime('%a %d %b')
                components_df = components_df.drop('date', axis=1)
                
                # Rename columns
                components_df.columns = [
                    'Date', 'Seasonal Base', 'Day of Week', 'Monthly', 
                    'Holiday', 'Growth', 'Prebooking', 'Final Prediction'
                ]
                
                st.dataframe(
                    components_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Explanation
                with st.expander("📖 Understanding the Components"):
                    st.markdown("""
                    **How predictions are calculated:**
                    
                    1. **Seasonal Base**: Historical median demand for this day of year
                    2. **Day of Week**: Multiplier based on weekday patterns (Tue-Thu typically higher)
                    3. **Monthly**: Seasonal adjustment by month
                    4. **Holiday**: UK bank holiday effects (bookend pattern for assisted travel)
                    5. **Growth**: Year-over-year growth projection
                    6. **Prebooking**: Adjustment based on current booking levels
                    7. **Final = Base × DoW × Monthly × Holiday × Growth × Prebooking**
                    
                    **Prebooking Logic:**
                    - If prebookings indicate **lower** demand → Flag only (no adjustment)
                    - If prebookings indicate **higher** demand:
                      - Within confidence interval → Update prediction
                      - Outside confidence interval → Flag only
                    """)
            
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
            
            # Export functionality
            st.header("📥 Export Predictions")
            
            # Prepare export data
            export_df = predictions_df[[
                'date', 'base_prediction', 'final_prediction', 'lower_bound', 
                'upper_bound', 'holiday_name', 'prebooking_flag'
            ]].copy()
            export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')
            export_df.columns = ['Date', 'Base_Prediction', 'Final_Prediction', 
                                'Lower_Bound', 'Upper_Bound', 'Holiday', 'Prebooking_Note']
            
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name=f"integrated_forecast_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
        else:
            st.warning("⚠️ No data loaded yet")
            
            # Provide clear instructions
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### 📤 Option 1: Upload Files Now
                
                **Quick Start:**
                1. Click "Upload Custom Data" in the sidebar
                2. Select your CSV files (2023, 2024, 2025)
                3. Click "Process Uploaded Data"
                
                This is the simplest method if you have the files ready.
                """)
            
            with col2:
                st.markdown("""
                ### 🔗 Option 2: OneDrive Setup
                
                **For automatic loading:**
                1. Get OneDrive share links for your CSV files
                2. Configure in Streamlit Secrets
                3. Files will load automatically on app start
                
                See instructions below for setup details.
                """)
            
            with st.expander("📚 Complete OneDrive Setup Instructions"):
                st.markdown("""
                ## Setting Up OneDrive for Automatic Data Loading
                
                ### Step 1: Prepare Your Files
                - Make sure your files are in **CSV format** (not Excel .xlsx)
                - Files should be named like: `2023 Database.csv`, `2024 Database.csv`, etc.
                
                ### Step 2: Upload to OneDrive
                1. Go to [OneDrive](https://onedrive.com)
                2. Upload your CSV files
                3. Wait for upload to complete
                
                ### Step 3: Get Share Links
                
                **Method A: Standard Share Link**
                1. Right-click on a CSV file
                2. Select "Share"
                3. Click "Anyone with the link can view"
                4. Click "Copy link"
                
                **Method B: Direct Download Link (More Reliable)**
                1. Click on the CSV file to open it
                2. Click the "Download" button at the top
                3. When download starts, right-click the download
                4. Select "Copy download link"
                
                ### Step 4: Configure Streamlit Secrets
                1. Go to your Streamlit app dashboard
                2. Click on your app → Settings → Secrets
                3. Add this configuration:
                
                ```toml
                [data_sources.onedrive]
                "2023 Database.csv" = "https://1drv.ms/x/s!YourLinkHere"
                "2024 Database.csv" = "https://1drv.ms/x/s!YourLinkHere"
                "2025 Database.csv" = "https://1drv.ms/x/s!YourLinkHere"
                ```
                
                4. Click "Save"
                5. Your app will restart and load the files automatically
                
                ### Troubleshooting OneDrive Issues
                
                **If downloads fail:**
                - ❌ Don't use folder links - share individual files
                - ❌ Don't use Excel files - convert to CSV first
                - ✅ Make sure permission is "Anyone with the link can view"
                - ✅ Try Method B (direct download link) if Method A fails
                - ✅ Test your link in an incognito browser window
                
                **Common OneDrive Link Formats:**
                - Personal: `https://1drv.ms/x/s!AbCdEf...`
                - Business: `https://companyname-my.sharepoint.com/...`
                
                **If OneDrive still doesn't work:**
                Consider using the manual upload option instead - it's more reliable
                and your data stays completely private.
                """)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        with st.expander("🔍 Error Details"):
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
