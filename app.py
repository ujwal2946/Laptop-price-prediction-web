import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings
import os
import time
from datetime import datetime
from streamlit_lottie import st_lottie
import requests
import plotly.graph_objects as go
import json

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Laptop Price Predictor", layout="wide", page_icon="💻")

# ----------------- LOTTIE LOADER ----------------- #
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_predict = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_3vbOcw.json")
lottie_history = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_5tkzkblw.json")

# ----------------- PERSISTENT STORAGE ----------------- #
HISTORY_FILE = "laptop_history.json"
FAVORITES_FILE = "laptop_favorites.json"

def load_history():
    """Load history from JSON file"""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.sidebar.warning(f"Could not load history: {e}")
    return []

def save_history():
    """Save history to JSON file"""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(st.session_state["history"], f, indent=2)
    except Exception as e:
        st.sidebar.warning(f"Could not save history: {e}")

def load_favorites():
    """Load favorites from JSON file"""
    try:
        if os.path.exists(FAVORITES_FILE):
            with open(FAVORITES_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.sidebar.warning(f"Could not load favorites: {e}")
    return []

def save_favorites():
    """Save favorites to JSON file"""
    try:
        with open(FAVORITES_FILE, 'w') as f:
            json.dump(st.session_state["favorites"], f, indent=2)
    except Exception as e:
        st.sidebar.warning(f"Could not save favorites: {e}")

# ----------------- MODEL LOADING ----------------- #
model_path = "laptop_price_model.pkl"
if not os.path.exists(model_path):
    st.warning("⚠️ Model file 'laptop_price_model.pkl' not found. Using enhanced fallback prediction.")
    class FallbackModel:
        def predict(self, X):
            if len(X.shape) == 1:
                processor, ram, storage = X
            else:
                processor, ram, storage = X[0]
            return self.calculate_enhanced_price(processor, ram, storage)
        
        def calculate_enhanced_price(self, processor, ram, storage):
            base_price = 45000
            processor_impact = (processor - 2.5) * 12000
            ram_impact = (ram - 8) * 1500
            storage_impact = (storage - 512) / 512 * 8000
            final_price = base_price + processor_impact + ram_impact + storage_impact
            
            if processor >= 4.0: final_price += 10000
            if ram >= 32: final_price += 8000
            if storage >= 1024: final_price += 5000
                
            return max(15000, final_price)
    
    model = FallbackModel()
    model_name = "Enhanced Fallback Model"
else:
    try:
        model = joblib.load(model_path)
        model_name = type(model).__name__
        st.sidebar.success(f"✅ Model loaded: {model_name}")
    except Exception as e:
        st.warning(f"⚠️ Error loading model: {e}. Using fallback.")
        class FallbackModel:
            def predict(self, X):
                if len(X.shape) == 1:
                    processor, ram, storage = X
                else:
                    processor, ram, storage = X[0]
                return self.calculate_enhanced_price(processor, ram, storage)
            
            def calculate_enhanced_price(self, processor, ram, storage):
                base_price = 45000
                processor_impact = (processor - 2.5) * 12000
                ram_impact = (ram - 8) * 1500
                storage_impact = (storage - 512) / 512 * 8000
                final_price = base_price + processor_impact + ram_impact + storage_impact
                
                if processor >= 4.0: final_price += 10000
                if ram >= 32: final_price += 8000
                if storage >= 1024: final_price += 5000
                    
                return max(15000, final_price)
        
        model = FallbackModel()
        model_name = "Enhanced Fallback Model"

# ----------------- SESSION STATE ----------------- #
if "history" not in st.session_state:
    st.session_state["history"] = load_history()
if "show_history" not in st.session_state:
    st.session_state["show_history"] = False
if "favorites" not in st.session_state:
    st.session_state["favorites"] = load_favorites()
if "prediction_made" not in st.session_state:
    st.session_state["prediction_made"] = False
if "show_analysis" not in st.session_state:
    st.session_state["show_analysis"] = None
if "selected_component" not in st.session_state:
    st.session_state["selected_component"] = "processor"

# >>> SANITIZE HISTORY <<<
def sanitize_history():
    required_keys = {"Processor", "RAM", "Storage", "Predicted Price"}
    cleaned = []
    for entry in st.session_state["history"]:
        if isinstance(entry, dict) and required_keys.issubset(entry.keys()):
            try:
                cleaned_entry = {
                    "Processor": float(entry["Processor"]),
                    "RAM": int(entry["RAM"]),
                    "Storage": int(entry["Storage"]),
                    "Predicted Price": float(entry["Predicted Price"])
                }
                if "Timestamp" in entry:
                    cleaned_entry["Timestamp"] = entry["Timestamp"]
                if "Laptop Profile" in entry:
                    cleaned_entry["Laptop Profile"] = entry["Laptop Profile"]
                if "id" in entry:
                    cleaned_entry["id"] = entry["id"]
                else:
                    cleaned_entry["id"] = f"{int(time.time())}_{len(cleaned)}"
                cleaned.append(cleaned_entry)
            except (ValueError, TypeError, KeyError):
                continue
    st.session_state["history"] = cleaned

sanitize_history()

# ----------------- HELPER FUNCTIONS ----------------- #
def generate_unique_id():
    return f"{int(time.time())}_{len(st.session_state['history'])}"

def glow_color(price):
    if price < 30000: return "#4caf50"
    elif price < 60000: return "#2196f3"
    elif price < 100000: return "#ffa500"
    else: return "#ff4b4b"

def feedback_text(price):
    if price < 30000: return "💰 Budget Friendly"
    elif price < 60000: return "⚖️ Mid-Range Value"
    elif price < 100000: return "🚀 Premium Performance"
    else: return "💎 High-End Powerhouse"

def get_laptop_profile(processor, ram, storage):
    if ram >= 32 or storage >= 2048 or processor >= 5.0:
        return "🔥 Gaming/Workstation"
    elif ram >= 16 and storage >= 1024:
        return "💼 Premium Productivity"
    elif ram >= 8 and 512 <= storage <= 1024:
        return "📚 Student/Office"
    else:
        return "📱 Basic Usage"

def get_upgrade_tips(price, processor, ram, storage):
    tips = []
    if ram < 8:
        tips.append("💾 **Upgrade to 8GB+ RAM** for smoother multitasking")
    if storage < 512:
        tips.append("💽 **Get 512GB+ SSD** for modern apps & OS")
    if processor < 2.0:
        tips.append("⚡ **Consider 2.0+ GHz CPU** for better performance")
    if price > 80000 and ram < 16:
        tips.append("💡 **16GB+ RAM expected** at this price range")
    if not tips:
        tips.append("✅ **Great configuration!** Good value for money")
    return tips

def display_countup_price(price):
    color = glow_color(price)
    feedback = feedback_text(price)
    placeholder = st.empty()
    step = max(100, int(price / 50))
    for i in range(0, int(price) + 1, step):
        html_content = f"""
        <div style='text-align:center; padding:30px; margin-top:10px; border-radius:15px;
                    background: linear-gradient(135deg, #1f1f2e, #2e2e3e);
                    box-shadow: 0 0 20px {color}, 0 0 40px {color};
                    color:white; position:relative; min-height:120px'>
            <h1 style='color:{color}; text-shadow: 0 0 15px {color}, 0 0 30px {color}; font-size:48px; font-weight:bold'>
                ₹{i:,.0f}
            </h1>
            <h3 style='color:{color}; margin-top:10px;'>{feedback}</h3>
        </div>
        """
        placeholder.markdown(html_content, unsafe_allow_html=True)
        time.sleep(0.01)

def show_dashboard(processor, ram, storage):
    dashboard_html = ""
    inputs = [
        ("⚡ Processor", f"{processor} GHz"),
        ("💾 RAM", f"{ram} GB"),
        ("💽 Storage", f"{storage} GB")
    ]
    for label, value in inputs:
        dashboard_html += f"""
        <div style='background:#2e2e3e; border-radius:12px; padding:10px; margin-bottom:8px;
                    box-shadow: 0 0 10px #00b4d8, 0 0 15px #00b4d8; text-align:center;'>
            <h5 style='color:#00b4d8; margin:0'>{label}</h5>
            <p style='font-size:18px; color:white; font-weight:bold; margin:0'>{value}</p>
        </div>
        """
    st.markdown(dashboard_html, unsafe_allow_html=True)

def add_timestamp_to_history():
    if st.session_state["history"]:
        st.session_state["history"][-1]["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["history"][-1]["Laptop Profile"] = get_laptop_profile(
            st.session_state["history"][-1]["Processor"],
            st.session_state["history"][-1]["RAM"],
            st.session_state["history"][-1]["Storage"]
        )
        save_history()

def delete_prediction(prediction_id):
    st.session_state["history"] = [entry for entry in st.session_state["history"] if entry.get("id") != prediction_id]
    st.session_state["favorites"] = [fav_id for fav_id in st.session_state["favorites"] if fav_id != prediction_id]
    save_history()
    save_favorites()
    st.success("✅ Prediction deleted!")
    st.rerun()

def clear_all_history():
    if st.session_state["history"]:
        st.session_state["history"] = []
        st.session_state["favorites"] = []
        save_history()
        save_favorites()
        st.success("🗑️ All history cleared!")
        st.rerun()
    else:
        st.info("📭 No history to clear!")

# ----------------- ENHANCED ANALYSIS FUNCTIONS ----------------- #
def create_component_analysis(base_processor, base_ram, base_storage, component):
    """Create analysis for selected component"""
    
    if component == "processor":
        values = np.linspace(1.0, 6.0, 15)
        prices = []
        for val in values:
            price = calculate_enhanced_price(val, base_ram, base_storage)
            prices.append(price)
        current_value = base_processor
        unit = "GHz"
        color = "#00b4d8"
        icon = "⚡"
        
    elif component == "ram":
        values = np.arange(4, 65, 4)
        prices = []
        for val in values:
            price = calculate_enhanced_price(base_processor, val, base_storage)
            prices.append(price)
        current_value = base_ram
        unit = "GB"
        color = "#4caf50"
        icon = "💾"
        
    else:  # storage
        values = np.linspace(128, 2048, 12)
        prices = []
        for val in values:
            price = calculate_enhanced_price(base_processor, base_ram, val)
            prices.append(price)
        current_value = base_storage
        unit = "GB"
        color = "#ffa500"
        icon = "💽"
    
    return {
        "values": values,
        "prices": prices,
        "current_value": current_value,
        "unit": unit,
        "color": color,
        "icon": icon
    }

def get_performance_rating(processor, ram, storage):
    """Get simple performance rating"""
    score = (processor/6.0 * 0.4) + (ram/64 * 0.35) + (storage/2048 * 0.25)
    
    if score >= 0.8:
        return "🎯 Excellent", "Perfect for gaming and professional work", "#00ff00"
    elif score >= 0.6:
        return "👍 Great", "Ideal for most users and multitasking", "#4caf50"
    elif score >= 0.4:
        return "✅ Good", "Good for everyday tasks and office work", "#2196f3"
    else:
        return "💡 Basic", "Suitable for browsing and light tasks", "#ff9800"

def get_cost_effectiveness(price, processor, ram, storage):
    """Evaluate if the configuration offers good value"""
    performance_score = (processor/6.0 * 0.4) + (ram/64 * 0.35) + (storage/2048 * 0.25)
    value_score = performance_score / (price / 100000)  # Normalize price
    
    if value_score > 1.5:
        return "💰 Excellent Value", "Great performance for the price!", "#4caf50"
    elif value_score > 1.0:
        return "💡 Good Value", "Reasonable price for the specs", "#2196f3"
    else:
        return "⚖️ Fair Value", "Consider optimizing your configuration", "#ff9800"

def show_enhanced_price_analysis(processor, ram, storage, price, config_index=None):
    """Show enhanced price analysis with selectable components"""
    
    st.markdown("---")
    
    if config_index is not None:
        st.markdown(f"## 📊 Analysis - Configuration {config_index + 1}")
    else:
        st.markdown("## 📊 Detailed Analysis")
    
    # Quick Overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        profile = get_laptop_profile(processor, ram, storage)
        st.metric("💻 Laptop Type", profile)
    
    with col2:
        rating, desc, color = get_performance_rating(processor, ram, storage)
        st.markdown(f"""
        <div style='background: {color}20; padding: 15px; border-radius: 10px; border-left: 4px solid {color};'>
            <div style='font-size: 16px; font-weight: bold; color: {color};'>{rating}</div>
            <div style='font-size: 12px; color: #ccc;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        value, value_desc, value_color = get_cost_effectiveness(price, processor, ram, storage)
        st.markdown(f"""
        <div style='background: {value_color}20; padding: 15px; border-radius: 10px; border-left: 4px solid {value_color};'>
            <div style='font-size: 16px; font-weight: bold; color: {value_color};'>{value}</div>
            <div style='font-size: 12px; color: #ccc;'>{value_desc}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Component Selector
    st.markdown("### 📈 Price Impact Analysis")
    
    # Component selection buttons with unique keys
    col1, col2, col3 = st.columns(3)
    
    with col1:
        button_key = f"processor_btn_{config_index if config_index is not None else 'current'}"
        if st.button("⚡ Processor", use_container_width=True, 
                    type="primary" if st.session_state["selected_component"] == "processor" else "secondary",
                    key=button_key):
            st.session_state["selected_component"] = "processor"
            st.rerun()
    
    with col2:
        button_key = f"ram_btn_{config_index if config_index is not None else 'current'}"
        if st.button("💾 RAM", use_container_width=True,
                    type="primary" if st.session_state["selected_component"] == "ram" else "secondary",
                    key=button_key):
            st.session_state["selected_component"] = "ram"
            st.rerun()
    
    with col3:
        button_key = f"storage_btn_{config_index if config_index is not None else 'current'}"
        if st.button("💽 Storage", use_container_width=True,
                    type="primary" if st.session_state["selected_component"] == "storage" else "secondary",
                    key=button_key):
            st.session_state["selected_component"] = "storage"
            st.rerun()
    
    # Get analysis for selected component
    component_data = create_component_analysis(processor, ram, storage, st.session_state["selected_component"])
    
    # Display component analysis
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create chart for selected component
        fig = go.Figure()
        
        # Main price trend
        fig.add_trace(go.Scatter(
            x=component_data["values"],
            y=component_data["prices"],
            mode='lines+markers',
            name='Price Trend',
            line=dict(color=component_data["color"], width=4),
            marker=dict(size=6, color=component_data["color"])
        ))
        
        # Current configuration marker
        fig.add_trace(go.Scatter(
            x=[component_data["current_value"]],
            y=[price],
            mode='markers',
            name='Your Configuration',
            marker=dict(size=16, color='#ff4b4b', symbol='star', 
                       line=dict(width=2, color='white'))
        ))
        
        fig.update_layout(
            title=f"{component_data['icon']} How {st.session_state['selected_component'].title()} Affects Price",
            xaxis_title=f"{st.session_state['selected_component'].title()} ({component_data['unit']})",
            yaxis_title="Predicted Price (₹)",
            template="plotly_dark",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f"### {component_data['icon']} Key Insights")
        
        # Calculate price differences
        current_idx = np.abs(component_data["values"] - component_data["current_value"]).argmin()
        
        if current_idx > 0:
            lower_price = component_data["prices"][current_idx-1]
            price_diff_lower = price - lower_price
            st.metric(
                f"⬇️ Lower tier", 
                f"₹{lower_price:,.0f}",
                f"Save ₹{price_diff_lower:,.0f}"
            )
        
        if current_idx < len(component_data["values"]) - 1:
            higher_price = component_data["prices"][current_idx+1]
            price_diff_higher = higher_price - price
            st.metric(
                f"⬆️ Higher tier", 
                f"₹{higher_price:,.0f}",
                f"+₹{price_diff_higher:,.0f}"
            )
        
        # Component-specific insights
        if st.session_state["selected_component"] == "processor":
            st.info("""
            **💡 Processor Insights:**
            - Each **+1 GHz** ≈ **₹12,000**
            - **4.0+ GHz** adds premium bonus
            - **Sweet spot**: 2.5-3.5 GHz
            """)
        
        elif st.session_state["selected_component"] == "ram":
            st.info("""
            **💡 RAM Insights:**
            - Each **+4 GB** ≈ **₹6,000**
            - **32GB+** adds premium bonus
            - **Recommended**: 8-16 GB
            """)
        
        else:  # storage
            st.info("""
            **💡 Storage Insights:**
            - Each **+512 GB** ≈ **₹8,000**
            - **1TB+** adds premium bonus
            - **Sweet spot**: 512GB-1TB
            """)
    
    # Quick Tips
    st.markdown("### 💡 Quick Tips")
    
    tips = get_upgrade_tips(price, processor, ram, storage)
    for i, tip in enumerate(tips):
        bg_color = "#2e2e3e" if i % 2 == 0 else "#1f1f2e"
        st.markdown(f"""
        <div style='background: {bg_color}; padding: 12px; border-radius: 8px; margin: 5px 0; border-left: 4px solid #00b4d8;'>
            <div style='font-size: 14px;'>{tip}</div>
        </div>
        """, unsafe_allow_html=True)

# ----------------- ENHANCED PRICE CALCULATION ----------------- #
def calculate_enhanced_price(processor, ram, storage):
    try:
        base_price = 45000
        processor_impact = (processor - 2.5) * 12000
        ram_impact = (ram - 8) * 1500
        storage_impact = (storage - 512) / 512 * 8000
        final_price = base_price + processor_impact + ram_impact + storage_impact
        
        if processor >= 4.0: final_price += 10000
        if ram >= 32: final_price += 8000
        if storage >= 1024: final_price += 5000
            
        return max(15000, final_price)
    
    except Exception as e:
        base_price = 45000
        processor_impact = (processor - 2.5) * 12000
        ram_impact = (ram - 8) * 1500
        storage_impact = (storage - 512) / 512 * 8000
        final_price = base_price + processor_impact + ram_impact + storage_impact
        
        if processor >= 4.0: final_price += 10000
        if ram >= 32: final_price += 8000
        if storage >= 1024: final_price += 5000
            
        return max(15000, final_price)

# ----------------- SIMPLIFIED COMPARISON FUNCTIONS ----------------- #
def show_comparison_tool(df):
    st.markdown("### 🔍 Quick Compare")
    
    if len(df) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            config1 = st.selectbox("Select first config", range(len(df)), 
                                   format_func=lambda x: f"Config {x+1}: ₹{df.iloc[x]['Predicted Price']:,.0f}",
                                   key="compare_config1")
        with col2:
            config2 = st.selectbox("Select second config", range(len(df)), 
                                   format_func=lambda x: f"Config {x+1}: ₹{df.iloc[x]['Predicted Price']:,.0f}",
                                   index=min(1, len(df)-1),
                                   key="compare_config2")
        
        if config1 != config2:
            row1, row2 = df.iloc[config1], df.iloc[config2]
            
            # Calculate differences
            price_diff = row2['Predicted Price'] - row1['Predicted Price']
            processor_diff = row2['Processor'] - row1['Processor']
            ram_diff = row2['RAM'] - row1['RAM']
            storage_diff = row2['Storage'] - row1['Storage']
            
            # Quick Summary Cards
            st.markdown("#### 📋 Quick Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style='background: rgba(0, 180, 216, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #00b4d8; margin: 10px 0;'>
                    <h4 style='color: #00b4d8; margin: 0;'>Config {config1+1}</h4>
                    <p style='color: white; font-size: 24px; font-weight: bold; margin: 5px 0;'>₹{row1['Predicted Price']:,.0f}</p>
                    <p style='color: #ccc; margin: 2px 0;'>⚡ {row1['Processor']} GHz</p>
                    <p style='color: #ccc; margin: 2px 0;'>💾 {row1['RAM']} GB</p>
                    <p style='color: #ccc; margin: 2px 0;'>💽 {row1['Storage']} GB</p>
                    <p style='color: #888; margin: 5px 0 0 0;'>{get_laptop_profile(row1['Processor'], row1['RAM'], row1['Storage'])}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='background: rgba(76, 175, 80, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #4caf50; margin: 10px 0;'>
                    <h4 style='color: #4caf50; margin: 0;'>Config {config2+1}</h4>
                    <p style='color: white; font-size: 24px; font-weight: bold; margin: 5px 0;'>₹{row2['Predicted Price']:,.0f}</p>
                    <p style='color: #ccc; margin: 2px 0;'>⚡ {row2['Processor']} GHz</p>
                    <p style='color: #ccc; margin: 2px 0;'>💾 {row2['RAM']} GB</p>
                    <p style='color: #ccc; margin: 2px 0;'>💽 {row2['Storage']} GB</p>
                    <p style='color: #888; margin: 5px 0 0 0;'>{get_laptop_profile(row2['Processor'], row2['RAM'], row2['Storage'])}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Key Differences
            st.markdown("#### ⚖️ Key Differences")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                price_color = "inverse" if price_diff > 0 else "normal"
                st.metric("💰 Price", f"₹{abs(price_diff):,.0f}", f"{price_diff:+,.0f}", delta_color=price_color)
            
            with col2:
                st.metric("⚡ CPU", f"{abs(processor_diff):.1f} GHz", f"{processor_diff:+.1f} GHz")
            
            with col3:
                st.metric("💾 RAM", f"{abs(ram_diff)} GB", f"{ram_diff:+d} GB")
            
            with col4:
                st.metric("💽 Storage", f"{abs(storage_diff)} GB", f"{storage_diff:+d} GB")
            
            # Simple Recommendation
            st.markdown("#### 💡 Quick Take")
            
            # Calculate which is better value
            score1 = (row1['Processor']/6.0 * 0.4) + (row1['RAM']/64 * 0.35) + (row1['Storage']/2048 * 0.25)
            score2 = (row2['Processor']/6.0 * 0.4) + (row2['RAM']/64 * 0.35) + (row2['Storage']/2048 * 0.25)
            
            value1 = score1 / (row1['Predicted Price'] / 100000)
            value2 = score2 / (row2['Predicted Price'] / 100000)
            
            if price_diff > 0:
                # Config 2 is more expensive
                if value2 > value1:
                    st.success(f"**Config {config2+1} offers better value** - You get more performance for your money")
                else:
                    st.warning(f"**Config {config1+1} is better value** - Similar performance for less money")
            else:
                # Config 2 is cheaper
                if value2 > value1:
                    st.success(f"**Config {config2+1} is the clear winner** - Better performance at lower cost")
                else:
                    st.info(f"**Config {config1+1} has better performance** - Worth the extra cost")
            
            # When to choose which
            st.markdown("#### 🎯 When to Choose Which?")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style='background: rgba(0, 180, 216, 0.1); padding: 12px; border-radius: 8px; margin: 5px 0;'>
                    <h5 style='color: #00b4d8; margin: 0 0 8px 0;'>Choose Config {config1+1} if:</h5>
                    <p style='color: #ccc; margin: 2px 0; font-size: 14px;'>• You want to save ₹{abs(price_diff):,.0f}</p>
                    <p style='color: #ccc; margin: 2px 0; font-size: 14px;'>• {get_laptop_profile(row1['Processor'], row1['RAM'], row1['Storage'])} meets your needs</p>
                    <p style='color: #ccc; margin: 2px 0; font-size: 14px;'>• Budget is your main concern</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='background: rgba(76, 175, 80, 0.1); padding: 12px; border-radius: 8px; margin: 5px 0;'>
                    <h5 style='color: #4caf50; margin: 0 0 8px 0;'>Choose Config {config2+1} if:</h5>
                    <p style='color: #ccc; margin: 2px 0; font-size: 14px;'>• You need the extra performance</p>
                    <p style='color: #ccc; margin: 2px 0; font-size: 14px;'>• {get_laptop_profile(row2['Processor'], row2['RAM'], row2['Storage'])} fits your use case</p>
                    <p style='color: #ccc; margin: 2px 0; font-size: 14px;'>• Future-proofing is important</p>
                </div>
                """, unsafe_allow_html=True)
            
    else:
        st.warning("📊 Need at least 2 predictions to compare")

# ----------------- HISTORY DISPLAY ----------------- #
def show_enhanced_history_section():
    if st.session_state["history"]:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 15px; margin-bottom: 20px;'>
            <h2 style='color: white; text-align: center; margin: 0;'>📜 Prediction History</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Clear All History", use_container_width=True, type="secondary", key="clear_all_btn"):
                st.warning("Delete ALL history? This cannot be undone.")
                col_confirm, col_cancel = st.columns(2)
                with col_confirm:
                    if st.button("✅ Yes", use_container_width=True, key="confirm_clear"):
                        clear_all_history()
                with col_cancel:
                    if st.button("❌ Cancel", use_container_width=True, key="cancel_clear"):
                        st.rerun()
        
        df = pd.DataFrame(st.session_state["history"])
        tab1, tab2, tab3 = st.tabs(["📋 All Predictions", "⭐ Favorites", "🔍 Compare"])
        
        with tab1:
            show_predictions_table(df)
        with tab2:
            show_favorites_section()
        with tab3:
            show_comparison_tool(df)
    else:
        st.info("📭 No predictions yet. Make your first prediction!")

def show_predictions_table(df):
    if df.empty:
        st.info("No predictions to display")
        return
        
    for idx, row in df.iterrows():
        prediction_id = row.get('id', f"pred_{idx}")
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([3, 2, 1, 1, 1])
            with col1:
                profile = row.get('Laptop Profile') or get_laptop_profile(row['Processor'], row['RAM'], row['Storage'])
                timestamp = row.get('Timestamp', 'Unknown date')
                st.markdown(f"""
                <div style='padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px; margin: 5px 0;'>
                    <div style='color: #00b4d8; font-weight: bold;'>{profile}</div>
                    <div style='color: white; font-size: 12px;'>
                        ⚡ {row['Processor']}GHz • 💾 {row['RAM']}GB • 💽 {row['Storage']}GB
                    </div>
                    <div style='color: #888; font-size: 10px; margin-top: 5px;'>{timestamp}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                price = row['Predicted Price']
                price_color = glow_color(price)
                st.markdown(f"""
                <div style='text-align: center; padding: 10px;'>
                    <div style='color: {price_color}; font-weight: bold; font-size: 18px;'>
                        ₹{price:,.0f}
                    </div>
                    <div style='color: #888; font-size: 12px;'>{feedback_text(price)}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                is_favorite = prediction_id in st.session_state["favorites"]
                button_text = "🌟" if is_favorite else "⭐"
                if st.button(button_text, key=f"fav_{prediction_id}"):
                    if is_favorite:
                        st.session_state["favorites"].remove(prediction_id)
                    else:
                        st.session_state["favorites"].append(prediction_id)
                    save_favorites()
                    st.rerun()
            with col4:
                button_text = "❌ Close" if st.session_state["show_analysis"] == prediction_id else "📈 Analyze"
                if st.button(button_text, key=f"analyze_{prediction_id}"):
                    if st.session_state["show_analysis"] == prediction_id:
                        st.session_state["show_analysis"] = None
                    else:
                        st.session_state["show_analysis"] = prediction_id
                    st.rerun()
            with col5:
                if st.button("🗑️", key=f"delete_{prediction_id}"):
                    delete_prediction(prediction_id)
    
    if st.session_state["show_analysis"] is not None:
        analysis_id = st.session_state["show_analysis"]
        analysis_row = None
        analysis_idx = None
        
        for idx, row in df.iterrows():
            if row.get('id') == analysis_id:
                analysis_row = row
                analysis_idx = idx
                break
        
        if analysis_row is not None:
            show_enhanced_price_analysis(
                analysis_row['Processor'], 
                analysis_row['RAM'], 
                analysis_row['Storage'], 
                analysis_row['Predicted Price'],
                config_index=analysis_idx
            )
            
            if st.button("❌ Close Analysis", key="close_analysis_btn", use_container_width=True):
                st.session_state["show_analysis"] = None
                st.rerun()

def show_favorites_section():
    if st.session_state["favorites"]:
        st.markdown("### ⭐ Favorite Configurations")
        df = pd.DataFrame(st.session_state["history"])
        if df.empty:
            st.info("No favorites to display")
            return
            
        favorites_df = df[df['id'].isin(st.session_state["favorites"])]
        
        for idx, row in favorites_df.iterrows():
            prediction_id = row.get('id', f"pred_{idx}")
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                with col1:
                    profile = row.get('Laptop Profile') or get_laptop_profile(row['Processor'], row['RAM'], row['Storage'])
                    timestamp = row.get('Timestamp', 'Unknown date')
                    st.markdown(f"""
                    <div style='padding: 15px; background: rgba(255,215,0,0.1); border-radius: 10px; margin: 10px 0; border: 2px solid #FFD700;'>
                        <div style='color: #FFD700; font-weight: bold;'>{profile}</div>
                        <div style='color: white; font-size: 14px;'>
                            ⚡ {row['Processor']}GHz • 💾 {row['RAM']}GB • 💽 {row['Storage']}GB
                        </div>
                        <div style='color: #888; font-size: 10px; margin-top: 5px;'>{timestamp}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    price = row['Predicted Price']
                    price_color = glow_color(price)
                    st.markdown(f"""
                    <div style='text-align: center; padding: 15px;'>
                        <div style='color: {price_color}; font-weight: bold; font-size: 20px;'>
                            ₹{price:,.0f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    if st.button("🔄 Use", key=f"use_fav_{prediction_id}"):
                        st.session_state["last_prediction"] = {
                            "Processor": row['Processor'],
                            "RAM": row['RAM'],
                            "Storage": row['Storage'],
                            "Predicted Price": price
                        }
                        st.success("💡 Configuration loaded!")
                with col4:
                    if st.button("🗑️", key=f"delete_fav_{prediction_id}"):
                        delete_prediction(prediction_id)
    else:
        st.info("⭐ No favorites yet. Click the star icon to save.")

# ----------------- MAIN APP ----------------- #
st.markdown("<h1 style='text-align:center; color:#00b4d8;'>💻 Laptop Price Predictor</h1>", unsafe_allow_html=True)

st.sidebar.markdown(f"**Model:** {model_name}")
st.sidebar.markdown(f"**Total Predictions:** {len(st.session_state['history'])}")
st.sidebar.markdown(f"**Favorites:** {len(st.session_state['favorites'])}")

col_inputs, col_dashboard = st.columns([2, 1])

with col_inputs:
    st.markdown("<h3 style='color:#00b4d8;'>🔧 Enter Specifications</h3>", unsafe_allow_html=True)
    
    processor = st.slider("⚡ Processor (GHz)", 1.0, 6.0, 2.5, 0.1, key="processor_slider")
    ram = st.slider("💾 RAM (GB)", 4, 64, 8, 4, key="ram_slider")
    storage = st.slider("💽 Storage (GB)", 128, 2048, 512, 128, key="storage_slider")

    col1, col2, col3 = st.columns(3)
    
    if not st.session_state["prediction_made"]:
        predict_clicked = col1.button("🎯 Predict Price", use_container_width=True, type="primary", key="predict_btn")
    else:
        predict_again_clicked = col1.button("🔄 Predict Again", use_container_width=True, type="primary", key="predict_again_btn")
    
    history_text = "📜 Hide History" if st.session_state["show_history"] else "📜 View History"
    history_clicked = col2.button(history_text, use_container_width=True, key="history_btn")

with col_dashboard:
    st.markdown("<h3 style='color:#00b4d8;'>📊 Current Configuration</h3>", unsafe_allow_html=True)
    show_dashboard(processor, ram, storage)

# ----------------- PREDICTION ----------------- #
if 'predict_clicked' in locals() and predict_clicked:
    with st.spinner("🧠 Analyzing specs..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        progress_bar.empty()
        
        try:
            price = calculate_enhanced_price(processor, ram, storage)
            price = max(15000, round(float(price), -2))
            
            new_prediction = {
                "id": generate_unique_id(),
                "Processor": processor,
                "RAM": ram,
                "Storage": storage,
                "Predicted Price": price,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Laptop Profile": get_laptop_profile(processor, ram, storage)
            }
            
            st.session_state["history"].append(new_prediction)
            save_history()
            
            st.session_state["prediction_made"] = True
            
            display_countup_price(price)
            if lottie_predict:
                st_lottie(lottie_predict, height=200, key="predict")
            
            st.balloons()
            
            # Show enhanced analysis
            show_enhanced_price_analysis(processor, ram, storage, price)
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")

if 'predict_again_clicked' in locals() and predict_again_clicked:
    st.session_state["prediction_made"] = False
    st.session_state["show_analysis"] = None
    st.rerun()

if history_clicked:
    st.session_state["show_history"] = not st.session_state["show_history"]
    st.rerun()

if st.session_state["show_history"]:
    show_enhanced_history_section()

# ----------------- FOOTER ----------------- #
st.divider()
st.markdown("<div style='text-align:center; color:#888;'><p>👨‍💻 Developed by CH Ujwal Sree | 💻 Laptop Price Predictor</p><p><b>made using GridSearchCV Model</b></p></div>", unsafe_allow_html=True)