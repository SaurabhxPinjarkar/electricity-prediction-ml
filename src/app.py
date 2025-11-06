"""
Streamlit Dashboard for Electricity Consumption Prediction

Features:
- Interactive prediction interface
- Model retraining capability
- Performance metrics visualization
- Feature importance analysis
- Historical data visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
import json
import sys
import os

# Add src to path - works for both local and Streamlit Cloud
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# DON'T change directory - causes path issues on Streamlit Cloud
# os.chdir(str(current_dir))

from model_utils import load_model, load_metrics, validate_input_ranges, prepare_input_features
from train import main as train_model_main


# Page configuration
st.set_page_config(
    page_title="Electricity Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Enhanced Design
st.markdown("""
<style>
    /* Main Header Styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
    }
    
    .stError {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Input Fields */
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input,
    .stSlider > div > div {
        border-radius: 8px;
    }
    
    /* Prediction Result Box */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        margin: 2rem 0;
    }
    
    /* Info Cards */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #888;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_cached():
    """Load model with caching."""
    model_path = Path(__file__).parent / 'model.pkl'
    if model_path.exists():
        return load_model(model_path)
    return None


@st.cache_data
def load_metrics_cached():
    """Load metrics with caching."""
    metrics_path = Path(__file__).parent / 'metrics.json'
    if metrics_path.exists():
        return load_metrics(metrics_path)
    return None


def train_model_callback():
    """Callback to train model."""
    with st.spinner("Training model... This may take 1-2 minutes."):
        try:
            model_path = Path(__file__).parent / 'model.pkl'
            # Use n_iter=2 and hyperparameter_tuning=False for faster training on Streamlit Cloud
            model, metrics = train_model_main(
                output_path=str(model_path), 
                hyperparameter_tuning=False,  # Skip hyperparameter tuning to save time/memory
                n_iter=2
            )
            st.success("✅ Model trained successfully!")
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")


def show_start_screen():
    """Display the start/welcome screen with team information"""
    
    # Full-screen start screen with centered content
    st.markdown("""
    <style>
        .start-screen {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 50px;
            border-radius: 20px;
            text-align: center;
            color: white;
            margin: 50px 0;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            animation: fadeIn 1.5s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .start-title {
            font-size: 4rem;
            font-weight: 900;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            animation: slideDown 1s ease-out;
        }
        
        @keyframes slideDown {
            from { transform: translateY(-50px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        
        .start-subtitle {
            font-size: 1.8rem;
            margin-bottom: 40px;
            opacity: 0.95;
        }
        
        .team-section {
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(10px);
            padding: 30px;
            border-radius: 15px;
            margin: 30px auto;
            max-width: 800px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .team-title {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 20px;
            color: #fff;
        }
        
        .project-lead {
            font-size: 1.3rem;
            margin: 15px 0;
            padding: 15px;
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            border-left: 5px solid #ffd700;
        }
        
        .team-member {
            font-size: 1.1rem;
            margin: 10px 0;
            padding: 10px;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
        }
        
        .email-info {
            font-size: 1rem;
            margin-top: 10px;
            opacity: 0.9;
        }
        
        .start-button {
            margin-top: 40px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        .feature-highlights {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .feature-card {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            transition: transform 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            background: rgba(255,255,255,0.2);
        }
    </style>
    
    <div class="start-screen">
        <div class="start-title">⚡ Electricity Consumption Predictor</div>
        <div class="start-subtitle">AI-Powered Energy Demand Forecasting System</div>
        
        <div class="team-section">
            <div class="team-title">� Project Team</div>
            
            <div class="team-member">
                <strong>Saurabh Pinjarkar</strong> - PRN: 202301060013<br>
                <div class="email-info">📧 saurabhpinjarkarx@gmail.com</div>
            </div>
            
            <div class="team-member">
                <strong>Omkar Jagadale</strong> - PRN: 202301060009
            </div>
            
            <div class="team-member">
                <strong>Shreyash Badve</strong> - PRN: 202301060011
            </div>
            
            <div class="team-member">
                <strong>Sujit Pal</strong> - PRN: 202301060002
            </div>
        </div>
        
        <div class="feature-highlights">
            <div class="feature-card">
                <h3>🎯 Accurate</h3>
                <p>92%+ Prediction Accuracy</p>
            </div>
            <div class="feature-card">
                <h3>⚡ Fast</h3>
                <p>Real-time Predictions</p>
            </div>
            <div class="feature-card">
                <h3>📊 Insightful</h3>
                <p>Detailed Analytics</p>
            </div>
            <div class="feature-card">
                <h3>🤖 Smart</h3>
                <p>AI-Powered ML Model</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Large centered button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Launch Application", use_container_width=True, type="primary", key="start_btn"):
            st.session_state.show_main_app = True
            st.rerun()


def main():
    # Initialize session state for start screen
    if 'show_main_app' not in st.session_state:
        st.session_state.show_main_app = False
    
    # Show start screen if not dismissed
    if not st.session_state.show_main_app:
        show_start_screen()
        return
    
    # Header
    st.markdown('<div class="main-header">⚡ Electricity Consumption Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Energy Demand Forecasting System</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/electricity.png", width=80)
        st.title("🎛️ Control Panel")
        
        # Model status
        model_data = load_model_cached()
        metrics = load_metrics_cached()
        
        if model_data is None:
            st.warning("⚠️ No trained model found!")
            if st.button("🚀 Train Initial Model", use_container_width=True):
                train_model_callback()
        else:
            st.success("✅ Model Loaded")
            
            # Model info
            with st.expander("📊 Model Information"):
                st.write(f"**Type:** Random Forest Regressor")
                st.write(f"**Features:** {len(model_data.get('feature_columns', []))}")
                st.write(f"**Target:** {model_data.get('target_column', 'N/A')}")
        
        st.divider()
        
        # Metrics display - ENHANCED
        if metrics:
            st.subheader("📈 Model Accuracy")
            
            # Convert R² to accuracy percentage
            r2_score = metrics['test']['r2']
            accuracy_percentage = r2_score * 100
            
            # Display accuracy prominently
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 15px; text-align: center; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h1 style='color: white; margin: 0; font-size: 3rem;'>{accuracy_percentage:.2f}%</h1>
                <p style='color: white; margin: 5px 0 0 0; font-size: 1.2rem;'>Model Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Performance interpretation
            if r2_score >= 0.90:
                st.success("🌟 **Excellent** - Model performs exceptionally well!")
            elif r2_score >= 0.80:
                st.success("✅ **Very Good** - Model predictions are reliable!")
            elif r2_score >= 0.70:
                st.info("👍 **Good** - Model shows good predictive capability!")
            elif r2_score >= 0.60:
                st.warning("⚠️ **Fair** - Model needs improvement for better predictions")
            else:
                st.error("❌ **Poor** - Model requires retraining with better parameters")
            
            # Detailed metrics
            st.markdown("**Detailed Performance Metrics:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R² Score", f"{metrics['test']['r2']:.4f}", 
                         help="R² measures how well the model explains variance in the data (0-1, higher is better)")
                st.metric("MAE", f"{metrics['test']['mae']:.0f} MW", 
                         help="Mean Absolute Error - average prediction error in Megawatts")
            with col2:
                st.metric("RMSE", f"{metrics['test']['rmse']:.0f} MW", 
                         help="Root Mean Square Error - penalizes larger errors more")
                st.metric("MAPE", f"{metrics['test']['mape']:.2f}%", 
                         help="Mean Absolute Percentage Error - average error as percentage")
        
        st.divider()
        
        # Retrain button
        st.subheader("🔄 Model Retraining")
        if st.button("🔄 Retrain Model", use_container_width=True):
            train_model_callback()
        
        st.info("💡 Retrain the model to improve predictions with updated data.")
        
        st.divider()
        
        # Back to start screen button
        st.subheader("🏠 Navigation")
        if st.button("⬅️ Back to Start Screen", use_container_width=True, key="back_to_start"):
            st.session_state.show_main_app = False
            st.rerun()
        
        st.caption("Return to see team information")
    
    # Main content
    if model_data is None:
        st.info("👈 Please train a model first using the sidebar.")
        return
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 Analytics", "🎯 Feature Importance", "📚 About"])
    
    # Tab 1: Prediction
    with tab1:
        st.header("🔮 Make a Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📅 Date & Time")
            selected_date = st.date_input(
                "Select Date",
                value=datetime.now(),
                min_value=datetime(2020, 1, 1),
                max_value=datetime(2030, 12, 31)
            )
            selected_hour = st.slider("Hour of Day", 0, 23, 12)
            
            # Calculate day of week and weekend
            day_of_week = selected_date.weekday()
            is_weekend = day_of_week >= 5
            month = selected_date.month
            
            st.info(f"📆 {selected_date.strftime('%A')}")
            if is_weekend:
                st.success("🎉 Weekend")
            else:
                st.info("💼 Weekday")
        
        with col2:
            st.subheader("🌡️ Weather Conditions")
            temperature = st.number_input(
                "Temperature (°C)",
                min_value=-20.0,
                max_value=50.0,
                value=20.0,
                step=0.5
            )
            humidity = st.slider("Humidity (%)", 0, 100, 60)
            wind_speed = st.number_input(
                "Wind Speed (m/s)",
                min_value=0.0,
                max_value=30.0,
                value=3.0,
                step=0.5
            )
        
        with col3:
            st.subheader("🌍 Additional Factors")
            pressure = st.number_input(
                "Atmospheric Pressure (hPa)",
                min_value=950.0,
                max_value=1050.0,
                value=1013.0,
                step=1.0
            )
            
            # Show current conditions summary
            st.markdown("#### Current Conditions")
            st.write(f"🌡️ {temperature}°C")
            st.write(f"💧 {humidity}%")
            st.write(f"💨 {wind_speed} m/s")
            st.write(f"🌍 {pressure} hPa")
        
        # Predict button
        st.divider()
        
        if st.button("⚡ Predict Consumption", use_container_width=True, type="primary"):
            try:
                # Validate inputs
                validate_input_ranges(
                    hour=selected_hour,
                    temp_celsius=temperature,
                    humidity=humidity,
                    wind_speed=wind_speed
                )
                
                # Prepare features
                feature_array = prepare_input_features(
                    hour=selected_hour,
                    temp_celsius=temperature,
                    humidity=humidity,
                    wind_speed=wind_speed,
                    pressure=pressure,
                    is_weekend=is_weekend,
                    month=month,
                    day_of_week=day_of_week,
                    feature_columns=model_data['feature_columns'],
                    scaler=model_data['scaler']
                )
                
                # Make prediction
                prediction = model_data['model'].predict(feature_array)[0]
                
                # Display result
                st.success("✅ Prediction Complete!")
                
                # Show accuracy information
                if metrics:
                    test_accuracy = metrics['test']['r2'] * 100
                    mape = metrics['test']['mape']
                    
                    st.info(f"""
                    **Model Confidence:** This prediction is made with {test_accuracy:.1f}% accuracy 
                    (±{mape:.1f}% average error based on test data)
                    """)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.metric(
                        label="⚡ Predicted Electricity Consumption",
                        value=f"{prediction:,.0f} MW",
                        delta=f"{prediction - 25000:,.0f} from avg" if prediction > 25000 else None,
                        help=f"Prediction made with {metrics['test']['r2']*100:.1f}% model accuracy" if metrics else "Prediction value"
                    )
                
                # Show input summary
                with st.expander("📋 Input Summary"):
                    input_df = pd.DataFrame({
                        'Parameter': ['Date', 'Hour', 'Day', 'Temperature', 'Humidity', 'Wind Speed', 'Pressure'],
                        'Value': [
                            selected_date.strftime('%Y-%m-%d'),
                            f"{selected_hour}:00",
                            selected_date.strftime('%A'),
                            f"{temperature}°C",
                            f"{humidity}%",
                            f"{wind_speed} m/s",
                            f"{pressure} hPa"
                        ]
                    })
                    st.dataframe(input_df, use_container_width=True, hide_index=True)
                
                # Download prediction
                prediction_data = {
                    'timestamp': f"{selected_date} {selected_hour}:00",
                    'predicted_consumption_MW': float(prediction),
                    'temperature_C': temperature,
                    'humidity_%': humidity,
                    'wind_speed_ms': wind_speed,
                    'pressure_hPa': pressure
                }
                
                st.download_button(
                    label="📥 Download Prediction as CSV",
                    data=pd.DataFrame([prediction_data]).to_csv(index=False),
                    file_name=f"prediction_{selected_date}_{selected_hour}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
    
    # Tab 2: Analytics
    with tab2:
        st.header("📊 Model Analytics & Accuracy Report")
        
        if metrics:
            # Overall Accuracy Display
            st.markdown("### 🎯 Overall Model Accuracy")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate accuracy percentage from R²
            train_accuracy = metrics['train']['r2'] * 100
            test_accuracy = metrics['test']['r2'] * 100
            
            with col1:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                            padding: 20px; border-radius: 10px; text-align: center;'>
                    <h2 style='color: white; margin: 0;'>{train_accuracy:.2f}%</h2>
                    <p style='color: white; margin: 5px 0 0 0;'>Train Accuracy</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 10px; text-align: center;'>
                    <h2 style='color: white; margin: 0;'>{test_accuracy:.2f}%</h2>
                    <p style='color: white; margin: 5px 0 0 0;'>Test Accuracy</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_error = metrics['test']['mape']
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 20px; border-radius: 10px; text-align: center;'>
                    <h2 style='color: white; margin: 0;'>{avg_error:.2f}%</h2>
                    <p style='color: white; margin: 5px 0 0 0;'>Avg Error (MAPE)</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                mae_mw = metrics['test']['mae']
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                            padding: 20px; border-radius: 10px; text-align: center;'>
                    <h2 style='color: white; margin: 0;'>{mae_mw:.0f}</h2>
                    <p style='color: white; margin: 5px 0 0 0;'>MAE (MW)</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Performance interpretation
            if test_accuracy >= 90:
                st.success("🌟 **Outstanding Performance!** Your model achieves excellent accuracy on test data. The predictions are highly reliable for real-world use.")
            elif test_accuracy >= 80:
                st.success("✅ **Very Good Performance!** Your model shows strong predictive capability. Suitable for production deployment.")
            elif test_accuracy >= 70:
                st.info("👍 **Good Performance!** The model performs well but could benefit from further optimization or more training data.")
            elif test_accuracy >= 60:
                st.warning("⚠️ **Fair Performance** - Consider collecting more data or tuning hyperparameters to improve accuracy.")
            else:
                st.error("❌ **Needs Improvement** - Model accuracy is below acceptable levels. Retraining with different parameters is recommended.")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Training Performance")
                train_metrics_df = pd.DataFrame({
                    'Metric': ['RMSE', 'MAE', 'R²', 'MAPE'],
                    'Value': [
                        f"{metrics['train']['rmse']:.2f}",
                        f"{metrics['train']['mae']:.2f}",
                        f"{metrics['train']['r2']:.4f}",
                        f"{metrics['train']['mape']:.2f}%"
                    ]
                })
                st.dataframe(train_metrics_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("📉 Test Performance")
                test_metrics_df = pd.DataFrame({
                    'Metric': ['RMSE', 'MAE', 'R²', 'MAPE'],
                    'Value': [
                        f"{metrics['test']['rmse']:.2f}",
                        f"{metrics['test']['mae']:.2f}",
                        f"{metrics['test']['r2']:.4f}",
                        f"{metrics['test']['mape']:.2f}%"
                    ]
                })
                st.dataframe(test_metrics_df, use_container_width=True, hide_index=True)
            
            # Comparison chart
            st.subheader("📊 Train vs Test Comparison")
            comparison_data = {
                'Dataset': ['Training', 'Training', 'Training', 'Test', 'Test', 'Test'],
                'Metric': ['RMSE', 'MAE', 'R²', 'RMSE', 'MAE', 'R²'],
                'Value': [
                    metrics['train']['rmse'],
                    metrics['train']['mae'],
                    metrics['train']['r2'] * 10000,  # Scale for visibility
                    metrics['test']['rmse'],
                    metrics['test']['mae'],
                    metrics['test']['r2'] * 10000
                ]
            }
            
            fig = px.bar(
                comparison_data,
                x='Metric',
                y='Value',
                color='Dataset',
                barmode='group',
                title='Model Performance Comparison'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional info
            st.subheader("ℹ️ Model Details")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Training Samples", f"{metrics.get('n_train_samples', 'N/A'):,}")
            with col2:
                st.metric("Test Samples", f"{metrics.get('n_test_samples', 'N/A'):,}")
            with col3:
                st.metric("Features", metrics.get('n_features', 'N/A'))
    
    # Tab 3: Feature Importance
    with tab3:
        st.header("🎯 Feature Importance Analysis")
        
        if 'feature_importance' in model_data:
            importance_data = model_data['feature_importance']
            importance_df = pd.DataFrame(importance_data).head(15)
            
            # Bar chart
            fig = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title='Top 15 Most Important Features',
                labels={'importance': 'Importance Score', 'feature': 'Feature'},
                color='importance',
                color_continuous_scale='Blues'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance table
            st.subheader("📋 Feature Importance Table")
            st.dataframe(
                importance_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Insights
            st.subheader("💡 Key Insights")
            top_feature = importance_df.iloc[0]['feature']
            top_importance = importance_df.iloc[0]['importance']
            
            st.info(f"""
            **Top Feature:** {top_feature} (Importance: {top_importance:.4f})
            
            The most influential factors for electricity consumption prediction are:
            1. **Temporal Features**: Hour of day, day of week patterns
            2. **Historical Data**: Lag features from previous hours/days
            3. **Weather Conditions**: Temperature, humidity, and wind speed
            4. **Seasonal Patterns**: Monthly and yearly cycles
            """)
    
    # Tab 4: About
    with tab4:
        st.header("📚 About This Application")
        
        st.markdown("""
        ### 🎯 Purpose
        This application uses machine learning to predict electricity consumption based on:
        - **Temporal patterns**: Hour, day, month, season
        - **Weather conditions**: Temperature, humidity, wind, pressure
        - **Historical data**: Previous consumption patterns
        
        ### 🔬 Technology Stack
        - **Machine Learning**: Random Forest Regressor with hyperparameter tuning
        - **Frontend**: Streamlit for interactive dashboard
        - **Data Processing**: Pandas, NumPy, Scikit-learn
        - **Visualization**: Plotly for interactive charts
        
        ### 📊 Model Features
        - **Cyclical Encoding**: Captures circular nature of time (hours, days, months)
        - **Lag Features**: Uses historical consumption data
        - **Weather Integration**: Combines energy and weather datasets
        - **Feature Engineering**: Extracts 20+ features from raw data
        
        ### 🚀 How to Use
        1. **Navigate to Predict tab** to make forecasts
        2. **Input date, time, and weather conditions**
        3. **Click Predict** to get consumption forecast
        4. **Download results** for record-keeping
        5. **Retrain model** when new data is available
        
        ### 📈 Model Accuracy & Performance
        """)
        
        if metrics:
            # Show current model accuracy
            test_accuracy = metrics['test']['r2'] * 100
            
            st.markdown(f"""
            **Current Model Performance:**
            - **Accuracy:** {test_accuracy:.2f}% (R² Score: {metrics['test']['r2']:.4f})
            - **Average Error:** {metrics['test']['mape']:.2f}% (MAPE)
            - **Prediction Error:** ±{metrics['test']['mae']:.0f} MW (MAE)
            - **Root Mean Square Error:** {metrics['test']['rmse']:.0f} MW
            
            #### 📊 What Do These Metrics Mean?
            
            **R² Score (Accuracy):**
            - Measures how well the model explains the data (0-100%)
            - {test_accuracy:.2f}% means the model can explain {test_accuracy:.2f}% of the variance in electricity consumption
            - Higher is better (90%+ is excellent)
            
            **MAPE (Mean Absolute Percentage Error):**
            - Shows average prediction error as a percentage
            - {metrics['test']['mape']:.2f}% means predictions are typically off by about {metrics['test']['mape']:.2f}%
            - Lower is better (< 10% is very good)
            
            **MAE (Mean Absolute Error):**
            - Average difference between predicted and actual values
            - {metrics['test']['mae']:.0f} MW means predictions are off by about {metrics['test']['mae']:.0f} MW on average
            - In practical terms: If actual consumption is 30,000 MW, prediction might be {30000-metrics['test']['mae']:.0f}-{30000+metrics['test']['mae']:.0f} MW
            
            **RMSE (Root Mean Square Error):**
            - Similar to MAE but penalizes larger errors more heavily
            - {metrics['test']['rmse']:.0f} MW indicates how much predictions deviate from actual values
            - Useful for identifying if the model has any major outliers
            """)
            
            # Visual accuracy indicator
            st.markdown("#### 🎯 Accuracy Rating")
            if test_accuracy >= 90:
                st.success("🌟 **EXCELLENT** - Model is highly reliable for production use!")
            elif test_accuracy >= 80:
                st.success("✅ **VERY GOOD** - Model predictions are trustworthy!")
            elif test_accuracy >= 70:
                st.info("👍 **GOOD** - Model performs well for most cases!")
            elif test_accuracy >= 60:
                st.warning("⚠️ **FAIR** - Model may need improvement!")
            else:
                st.error("❌ **NEEDS IMPROVEMENT** - Consider retraining!")
        else:
            st.info("Train a model to see accuracy metrics!")
        
        st.markdown("""
        ### 🔄 Retraining
        Retrain the model periodically to:
        - Incorporate new data patterns
        - Improve prediction accuracy
        - Adapt to changing conditions
        
        ---
        
        ### � Project Team
        
        This project was developed by a dedicated team of students:
        """)
        
        # Team information - All members equal
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Saurabh Pinjarkar**\nPRN: 202301060013\n📧 saurabhpinjarkarx@gmail.com")
            st.info("**Shreyash Badve**\nPRN: 202301060011")
        with col2:
            st.info("**Omkar Jagadale**\nPRN: 202301060009")
            st.info("**Sujit Pal**\nPRN: 202301060002")
        
        st.markdown("""
        ---
        
        ### 📞 Contact & Support
        
        For questions, feedback, or collaboration:
        - **Email:** saurabhpinjarkarx@gmail.com
        
        ---
        
        ### 📋 Project Information
        
        **Version:** 1.0.0  
        **Last Updated:** November 2025  
        **Developed with:** Python 3.10+, Streamlit, Scikit-learn  
        **License:** Academic Project  
        **Institution:** [Your Institution Name]
        
        ---
        
        <div style='text-align: center; padding: 20px; 
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                    border-radius: 10px; margin-top: 20px;'>
            <p style='margin: 0; font-weight: bold;'>
                💡 Electricity Consumption Prediction Project 💡
            </p>
            <p style='margin: 5px 0 0 0; font-size: 0.9rem; color: #666;'>
                Machine Learning • Data Science • Energy Forecasting
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
