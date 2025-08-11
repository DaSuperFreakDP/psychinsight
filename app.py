import streamlit as st
import pandas as pd
import numpy as np
from data_manager import DataManager
from statistical_analyzer import StatisticalAnalyzer
from ai_insights import AIInsights
from visualizations import Visualizations
import io

# Configure page
st.set_page_config(
    page_title="Statistical Analysis with AI Insights - Enhanced",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()
if 'statistical_analyzer' not in st.session_state:
    st.session_state.statistical_analyzer = StatisticalAnalyzer()
if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = AIInsights()
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = Visualizations()

def main():
    st.title("📊 Statistical Analysis with AI Insights")
    st.markdown("Upload your experimental data and get comprehensive statistical analysis with AI-powered insights.")
    
    # Debug marker to confirm new code is loaded
    st.sidebar.success("🔄 Enhanced Version Loaded - v2.0")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        
        # File upload section
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your data file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV or Excel files containing your experimental data"
        )
        
        
        # Initialize variables to avoid unbound errors
        data = None
        filtered_data = None
        column_descriptions = {}
        selected_tests = []
        run_analysis = False
        
        if uploaded_file is not None:
            try:
                data = st.session_state.data_manager.load_data(uploaded_file)
                st.success(f"✅ Data loaded successfully! Shape: {data.shape}")
                
                # Column descriptions section
                st.subheader("📝 Column Setup")
                st.markdown("Set display names and descriptions for your columns:")
                
                column_descriptions = {}
                column_display_names = {}
                
                # Create columns for layout
                desc_cols = st.columns(2)
                
                for i, col in enumerate(data.columns):
                    with desc_cols[i % 2]:
                        with st.expander(f"Column: {col}", expanded=False):
                            # Display name
                            display_name = st.text_input(
                                f"Display name for {col}:",
                                value=col,
                                key=f"display_{col}",
                                help="Friendly name to show in the interface"
                            )
                            column_display_names[col] = display_name
                            
                            # Description for AI
                            description = st.text_area(
                                f"Description for AI analysis:",
                                key=f"desc_{col}",
                                placeholder="Describe what this column measures, the test methodology, expected values, etc. This helps AI provide better insights.",
                                height=100
                            )
                            column_descriptions[col] = description
                
                st.session_state.data_manager.set_column_descriptions(column_descriptions)
                
                # Data filtering section
                st.subheader("🔍 Data Filtering")
                filtered_data = apply_filters(data)
                
                # Statistical test selection (moved to main area)
                
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                return
    
    # Main content area
    if uploaded_file is not None and data is not None and filtered_data is not None:
        
        # Display data preview and interactions
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Data Preview & Analysis")
            
            # Display selectable dataframe with selection capabilities
            edited_data = st.data_editor(
                filtered_data, 
                use_container_width=True, 
                key="data_editor",
                num_rows="dynamic",
                on_change=None
            )
            
            # Store edited data
            st.session_state.selected_data = edited_data
            
            # Cell selection for calculations
            st.subheader("📊 Selection-Based Calculations")
            
            calc_col1, calc_col2, calc_col3 = st.columns(3)
            
            with calc_col1:
                calc_column = st.selectbox(
                    "Select column for calculation:",
                    options=[col for col in filtered_data.columns if filtered_data[col].dtype in ['int64', 'float64']],
                    key="calc_column"
                )
            
            with calc_col2:
                start_row = 0
                end_row = 0
                if calc_column:
                    # Row selection for calculation
                    start_row = st.number_input("Start row:", min_value=0, max_value=len(filtered_data)-1, value=0, key="calc_start")
                    end_row = st.number_input("End row:", min_value=start_row, max_value=len(filtered_data)-1, value=min(4, len(filtered_data)-1), key="calc_end")
            
            with calc_col3:
                calc_type = st.selectbox(
                    "Calculation:",
                    ["Mean", "Sum", "Count", "Min", "Max", "Std Dev"],
                    key="calc_type"
                )
                
                if st.button("📊 Calculate Selection", type="primary"):
                    if calc_column and calc_column in filtered_data.columns:
                        selected_values = filtered_data[calc_column].iloc[start_row:end_row+1]
                        result = 0
                        
                        if calc_type == "Mean":
                            result = selected_values.mean()
                        elif calc_type == "Sum":
                            result = selected_values.sum()
                        elif calc_type == "Count":
                            result = len(selected_values.dropna())
                        elif calc_type == "Min":
                            result = selected_values.min()
                        elif calc_type == "Max":
                            result = selected_values.max()
                        elif calc_type == "Std Dev":
                            result = selected_values.std()
                        
                        st.success(f"**{calc_type} of {calc_column} (rows {start_row}-{end_row}):** {result:.3f}")
                        st.info(f"Selected {len(selected_values)} values from column '{calc_column}'")
            
            # Cell analysis section
            st.subheader("🔍 Cell Value Analysis")
            
            cell_col1, cell_col2, cell_col3 = st.columns(3)
            
            with cell_col1:
                analysis_column = st.selectbox(
                    "Select column:",
                    options=filtered_data.columns.tolist(),
                    key="cell_analysis_column"
                )
            
            with cell_col2:
                analysis_value = None
                if analysis_column in filtered_data.columns:
                    if filtered_data[analysis_column].dtype in ['object', 'string']:
                        analysis_value = st.selectbox(
                            "Select value:",
                            options=filtered_data[analysis_column].dropna().unique(),
                            key="cell_analysis_value"
                        )
                    else:
                        analysis_value = st.number_input(
                            f"Value to analyze:",
                            value=float(filtered_data[analysis_column].mean()) if not filtered_data[analysis_column].isna().all() else 0.0,
                            key="cell_analysis_number"
                        )
            
            with cell_col3:
                if st.button("🧠 AI Cell Insight", type="primary") and analysis_value is not None:
                    with st.spinner("Analyzing value..."):
                        try:
                            col_desc = column_descriptions.get(analysis_column, "No description provided")
                            
                            if filtered_data[analysis_column].dtype in ['int64', 'float64']:
                                column_stats = filtered_data[analysis_column].describe()
                                percentile = (filtered_data[analysis_column] < analysis_value).mean() * 100
                                
                                st.success(f"**AI Analysis for {analysis_column} = {analysis_value}:**")
                                st.write(f"- **Context**: {col_desc}")
                                st.write(f"- **Value Position**: {percentile:.1f}th percentile")
                                
                                # Enhanced assessment based on description
                                if "lower" in col_desc.lower() and ("better" in col_desc.lower() or "good" in col_desc.lower()):
                                    # Lower is better scenario
                                    if percentile < 25:
                                        st.write("- **Performance Assessment**: **EXCELLENT** - This is a low value, which is good performance based on your description")
                                    elif percentile > 75:
                                        st.write("- **Performance Assessment**: **POOR** - This is a high value, which indicates poor performance based on your description")
                                    else:
                                        st.write("- **Performance Assessment**: **AVERAGE** - This is a typical value for this measure")
                                elif "higher" in col_desc.lower() and ("better" in col_desc.lower() or "good" in col_desc.lower()):
                                    # Higher is better scenario
                                    if percentile > 75:
                                        st.write("- **Performance Assessment**: **EXCELLENT** - This is a high value, which is good performance based on your description")
                                    elif percentile < 25:
                                        st.write("- **Performance Assessment**: **POOR** - This is a low value, which indicates poor performance based on your description")
                                    else:
                                        st.write("- **Performance Assessment**: **AVERAGE** - This is a typical value for this measure")
                                else:
                                    # Standard assessment
                                    if percentile < 25:
                                        st.write("- **Assessment**: This is a **low** value for this measure")
                                    elif percentile > 75:
                                        st.write("- **Assessment**: This is a **high** value for this measure")
                                    else:
                                        st.write("- **Assessment**: This is a **typical** value for this measure")
                                
                                st.write(f"- **Statistical Range**: {column_stats['min']:.2f} to {column_stats['max']:.2f} (Mean: {column_stats['mean']:.2f})")
                                
                                # Add interpretation guidance
                                if col_desc and col_desc != "No description provided":
                                    st.write(f"- **Interpretation Guide**: {col_desc}")
                                    
                            else:
                                value_count = (filtered_data[analysis_column] == analysis_value).sum()
                                total_count = len(filtered_data)
                                percentage = (value_count / total_count) * 100
                                
                                st.success(f"**AI Analysis for {analysis_column} = {analysis_value}:**")
                                st.write(f"- **Context**: {col_desc}")
                                st.write(f"- **Frequency**: {value_count} out of {total_count} rows ({percentage:.1f}%)")
                                if percentage > 50:
                                    st.write("- **Assessment**: This is the **most common** value")
                                elif percentage > 25:
                                    st.write("- **Assessment**: This is a **common** value")
                                else:
                                    st.write("- **Assessment**: This is a **less common** value")
                                    
                        except Exception as e:
                            st.error(f"Error analyzing value: {str(e)}")
            
            # Statistical Analysis Section
            st.subheader("📊 Statistical Analysis Options")
            
            stat_col1, stat_col2 = st.columns([2, 1])
            
            with stat_col1:
                selected_tests = st.multiselect(
                    "Choose statistical tests:",
                    ["Descriptive Statistics", "T-Test", "ANOVA", "Regression", "Correlation"],
                    default=["Descriptive Statistics"],
                    key="stats_multiselect",
                    help="Select which statistical analyses to perform on your data"
                )
            
            with stat_col2:
                run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
            
            # Run statistical analysis
            if selected_tests and run_analysis:
                st.subheader("📈 Statistical Results")
                
                with st.spinner("Running statistical analysis..."):
                    results = {}
                    analysis_data = edited_data
                    
                    for test in selected_tests:
                        try:
                            if test == "Descriptive Statistics":
                                results[test] = st.session_state.statistical_analyzer.descriptive_stats(analysis_data)
                            elif test == "T-Test" and len(analysis_data.select_dtypes(include=[np.number]).columns) >= 2:
                                results[test] = st.session_state.statistical_analyzer.t_test(analysis_data)
                            elif test == "ANOVA" and len(analysis_data.select_dtypes(include=[np.number]).columns) >= 2:
                                results[test] = st.session_state.statistical_analyzer.anova_test(analysis_data)
                            elif test == "Regression" and len(analysis_data.select_dtypes(include=[np.number]).columns) >= 2:
                                results[test] = st.session_state.statistical_analyzer.regression_analysis(analysis_data)
                            elif test == "Correlation":
                                results[test] = st.session_state.statistical_analyzer.correlation_analysis(analysis_data)
                        except Exception as e:
                            st.error(f"Error running {test}: {str(e)}")
                
                # Display results with AI insights
                for test_name, result in results.items():
                    if result is not None:
                        with st.expander(f"📈 {test_name} Results", expanded=True):
                            # Display statistical results
                            display_statistical_results(test_name, result)
                            
                            # Add AI statistical interpretation
                            st.markdown("---")
                            st.markdown("### 🧠 AI Statistical Interpretation")
                            
                            with st.spinner("Generating detailed statistical insights..."):
                                try:
                                    stat_insight = generate_statistical_insight(test_name, result, analysis_data, column_descriptions)
                                    st.markdown(stat_insight)
                                except Exception as e:
                                    st.warning(f"Could not generate AI insight: {str(e)}")
                
                # Store results for AI insights
                st.session_state.analysis_results = results
                st.success("Statistical analysis completed!")
            
            elif selected_tests and not run_analysis:
                st.info("Click 'Run Analysis' button to see results")
            
            # Visualizations
            if len(filtered_data) > 0:
                st.subheader("📈 Data Visualizations")
                results = getattr(st.session_state, 'analysis_results', {})
                create_visualizations(filtered_data, results)
        
        with col2:
            st.subheader("🤖 AI Insights")
            
            # Check if we have analysis results to work with
            results = getattr(st.session_state, 'analysis_results', {})
            
            if results:
                if st.button("🧠 Generate AI Insights", type="primary", use_container_width=True):
                    with st.spinner("Analyzing your data with AI..."):
                        try:
                            insights = st.session_state.ai_insights.generate_insights(
                                filtered_data, 
                                results, 
                                column_descriptions
                            )
                            
                            # Store insights in session state
                            st.session_state.ai_insights_results = insights
                            
                            # Display insights with better formatting
                            st.success("AI Analysis Complete!")
                            
                        except Exception as e:
                            st.error(f"Error generating AI insights: {str(e)}")
                            st.info("Make sure you have set your OpenAI API key in the environment variables.")
            else:
                st.info("🔍 Run statistical analysis first to get AI insights")
            
            # Display stored insights if available
            if hasattr(st.session_state, 'ai_insights_results'):
                insights = st.session_state.ai_insights_results
                
                with st.container():
                    st.markdown("### 🔍 Key Findings")
                    st.markdown(insights.get("key_findings", "No key findings generated"))
                    
                    st.markdown("### 💡 Recommendations")
                    st.markdown(insights.get("recommendations", "No recommendations generated"))
                    
                    st.markdown("### 📊 Statistical Significance")
                    st.markdown(insights.get("significance", "No significance analysis generated"))
            
            # Export functionality
            st.subheader("💾 Export Results")
            
            results = getattr(st.session_state, 'analysis_results', {})
            if results:
                export_options = st.multiselect(
                    "Select what to include in export:",
                    ["Statistical Results", "AI Insights", "Data Summary", "Column Descriptions"],
                    default=["Statistical Results", "AI Insights"],
                    key="export_options"
                )
                
                export_format = st.selectbox(
                    "Export format:",
                    ["Text Report", "CSV Data", "JSON Results"],
                    key="export_format"
                )
                
                if st.button("📄 Generate Export", use_container_width=True):
                    try:
                        if export_format == "Text Report":
                            report = generate_report(filtered_data, results, column_descriptions, export_options)
                            st.download_button(
                                label="📥 Download Text Report",
                                data=report,
                                file_name="statistical_analysis_report.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        elif export_format == "CSV Data" and filtered_data is not None:
                            csv_data = filtered_data.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Filtered Data",
                                data=csv_data,
                                file_name="filtered_data.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        elif export_format == "JSON Results":
                            import json
                            json_data = json.dumps(results, indent=2, default=str)
                            st.download_button(
                                label="📥 Download JSON Results",
                                data=json_data,
                                file_name="analysis_results.json",
                                mime="application/json",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"Error generating export: {str(e)}")
            else:
                st.info("Run statistical analysis first to enable export options")
    
    else:
        # Show the layout even without data
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("👆 Please upload a data file to begin analysis.")
            
            # Show example of expected data format
            st.subheader("📋 Expected Data Format")
            example_data = pd.DataFrame({
                'Group': ['Control', 'Treatment', 'Control', 'Treatment'],
                'Score': [85.2, 92.1, 78.9, 88.7],
                'Age': [25, 30, 22, 28],
                'Gender': ['M', 'F', 'M', 'F']
            })
            st.dataframe(example_data)
            st.caption("Example: Your data should have clear column headers and can contain both numeric and categorical data.")
        
        with col2:
            st.subheader("🤖 AI Insights")
            st.info("Upload data and run statistical analysis to generate AI insights")
            
            st.subheader("💾 Export Results")
            st.info("Export options will appear after running analysis")
            
            # Show example of what filtering looks like
            st.subheader("📋 Example: Column Filtering")
            st.markdown("**Group Column Filter:**")
            st.checkbox("✓ Control (10 rows)", value=True, disabled=True)
            st.checkbox("✓ Treatment (11 rows)", value=True, disabled=True)
            st.caption("Example: Uncheck 'Control' to show only Treatment group")

def apply_filters(data):
    """Apply simple column-by-column filtering with checkboxes"""
    filtered_data = data.copy()
    original_count = len(data)
    
    st.markdown("**Filter Data by Column Values:**")
    st.caption("Uncheck values to hide rows containing those values")
    
    # Process each column for filtering
    for col in data.columns:
        if data[col].dtype in ['object', 'string', 'category']:
            # For categorical columns, show checkboxes
            unique_values = data[col].dropna().unique()
            
            if len(unique_values) <= 20:  # Only show filtering for manageable number of values
                st.markdown(f"**{col}:**")
                
                # Create columns for layout
                check_cols = st.columns(min(4, len(unique_values)))
                
                selected_values = []
                for i, value in enumerate(unique_values):
                    with check_cols[i % len(check_cols)]:
                        count = (data[col] == value).sum()
                        is_selected = st.checkbox(
                            f"{value} ({count})",
                            value=True,
                            key=f"show_{col}_{value}",
                            help=f"Show rows where {col} = {value}"
                        )
                        if is_selected:
                            selected_values.append(value)
                
                # Quick control buttons
                btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
                with btn_col1:
                    if st.button("All", key=f"select_all_{col}", help="Show all values"):
                        for value in unique_values:
                            st.session_state[f"show_{col}_{value}"] = True
                        st.rerun()
                
                with btn_col2:
                    if st.button("Clear", key=f"clear_all_{col}", help="Hide all values"):
                        for value in unique_values:
                            st.session_state[f"show_{col}_{value}"] = False
                        st.rerun()
                
                # Apply the filter
                if selected_values:
                    filtered_data = filtered_data[filtered_data[col].isin(selected_values)]
                else:
                    # If nothing selected, show empty dataframe
                    filtered_data = filtered_data.iloc[0:0]
                
                st.markdown("---")
        
        elif data[col].dtype in ['int64', 'float64']:
            # For numeric columns, show range slider
            if not data[col].isna().all():
                min_val, max_val = float(data[col].min()), float(data[col].max())
                
                if min_val != max_val:
                    st.markdown(f"**{col} Range:**")
                    selected_range = st.slider(
                        f"Range for {col}",
                        min_val, max_val, (min_val, max_val),
                        key=f"range_filter_{col}",
                        help=f"Filter {col} values between selected range"
                    )
                    
                    # Apply numeric filter
                    filtered_data = filtered_data[
                        (filtered_data[col] >= selected_range[0]) & 
                        (filtered_data[col] <= selected_range[1])
                    ]
                    
                    st.caption(f"Range: {selected_range[0]:.2f} to {selected_range[1]:.2f}")
                    st.markdown("---")
    
    # Show filtering summary
    filtered_count = len(filtered_data)
    hidden_count = original_count - filtered_count
    
    if hidden_count > 0:
        st.error(f"📊 Showing {filtered_count} of {original_count} rows ({hidden_count} rows hidden)")
    else:
        st.success(f"📊 Showing all {filtered_count} rows (no filters applied)")
    
    return filtered_data

def display_statistical_results(test_name, results):
    """Display statistical test results in a formatted way"""
    if test_name == "Descriptive Statistics":
        st.dataframe(results, use_container_width=True)
    elif test_name in ["T-Test", "ANOVA"]:
        for test_result in results:
            st.write(f"**{test_result['test_name']}**")
            st.write(f"- Statistic: {test_result['statistic']:.4f}")
            st.write(f"- P-value: {test_result['p_value']:.4f}")
            st.write(f"- Significant: {'Yes' if test_result['p_value'] < 0.05 else 'No'}")
            st.write("---")
    elif test_name == "Regression":
        st.write("**Regression Analysis Results**")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                st.write(f"- {key}: {value:.4f}")
            else:
                st.write(f"- {key}: {value}")
    elif test_name == "Correlation":
        st.write("**Correlation Matrix**")
        st.dataframe(results, use_container_width=True)

def create_visualizations(data, results):
    """Create and display visualizations"""
    viz_type = st.selectbox(
        "Select visualization type:",
        ["Histogram", "Scatter Plot", "Box Plot", "Correlation Heatmap"]
    )
    
    if viz_type == "Histogram":
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select column for histogram:", numeric_cols)
            fig = st.session_state.visualizations.create_histogram(data, selected_col)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Scatter Plot":
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            col1 = st.selectbox("X-axis:", numeric_cols, key="scatter_x")
            col2 = st.selectbox("Y-axis:", numeric_cols, key="scatter_y")
            if col1 != col2:
                fig = st.session_state.visualizations.create_scatter_plot(data, col1, col2)
                st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot":
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(numeric_cols) > 0:
            numeric_col = st.selectbox("Select numeric column:", numeric_cols)
            categorical_col = None
            if len(categorical_cols) > 0:
                categorical_col = st.selectbox("Group by (optional):", [None] + list(categorical_cols))
            fig = st.session_state.visualizations.create_box_plot(data, numeric_col, categorical_col)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Heatmap":
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            fig = st.session_state.visualizations.create_correlation_heatmap(data)
            st.plotly_chart(fig, use_container_width=True)

def generate_report(data, results, descriptions, export_options=None):
    """Generate a comprehensive analysis report"""
    if export_options is None:
        export_options = ["Statistical Results", "AI Insights", "Data Summary", "Column Descriptions"]
    
    report = io.StringIO()
    
    report.write("STATISTICAL ANALYSIS REPORT\n")
    report.write("=" * 50 + "\n\n")
    
    report.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.write(f"Data Shape: {data.shape[0]} rows × {data.shape[1]} columns\n\n")
    
    if "Data Summary" in export_options:
        report.write("DATA SUMMARY:\n")
        report.write(str(data.describe()))
        report.write("\n\n")
    
    return report.getvalue()

def generate_statistical_insight(test_name, result, data, column_descriptions):
    """Generate detailed AI insights for statistical results"""
    n_samples = len(data)
    
    if test_name == "Descriptive Statistics":
        insights = []
        insights.append("**Sample Size Assessment:**")
        if n_samples < 30:
            insights.append(f"- With only {n_samples} observations, this is a small sample that may limit generalizability and statistical power")
        elif n_samples < 100:
            insights.append(f"- Sample size of {n_samples} is moderate; results should be interpreted with caution")
        else:
            insights.append(f"- Sample size of {n_samples} provides good statistical power for most analyses")
        
        # Analyze distributions
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            insights.append("\n**Distribution Analysis:**")
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                skewness = data[col].skew()
                if abs(skewness) > 1:
                    insights.append(f"- {col}: Highly skewed distribution (skewness: {skewness:.2f}) - consider non-parametric tests")
                elif abs(skewness) > 0.5:
                    insights.append(f"- {col}: Moderately skewed distribution (skewness: {skewness:.2f})")
                else:
                    insights.append(f"- {col}: Approximately normal distribution (skewness: {skewness:.2f})")
        
        return "\n".join(insights)
    
    elif test_name == "T-Test" and isinstance(result, list):
        insights = []
        
        for test_result in result:
            if isinstance(test_result, dict) and 'p_value' in test_result:
                p_val = test_result['p_value']
                statistic = test_result.get('statistic', 0)
                test_type = test_result.get('test_name', 'T-Test')
                
                insights.append(f"**{test_type} Analysis:**")
                
                # P-value interpretation
                if p_val < 0.001:
                    insights.append(f"- Extremely strong evidence against null hypothesis (p = {p_val:.4f})")
                elif p_val < 0.01:
                    insights.append(f"- Very strong evidence against null hypothesis (p = {p_val:.3f})")
                elif p_val < 0.05:
                    insights.append(f"- Statistically significant result (p = {p_val:.3f})")
                elif p_val < 0.10:
                    insights.append(f"- Marginally significant result (p = {p_val:.3f}) - interpret with caution")
                else:
                    insights.append(f"- No significant difference found (p = {p_val:.3f})")
                
                # Statistical power considerations
                if n_samples < 20:
                    insights.append(f"- **Power Concern**: Small sample size (n={n_samples}) provides low statistical power, increasing risk of Type II error")
                elif n_samples < 50:
                    insights.append(f"- **Moderate Power**: Sample size (n={n_samples}) provides moderate power for detecting medium to large effects")
                else:
                    insights.append(f"- **Good Power**: Sample size (n={n_samples}) provides good power for detecting meaningful effects")
                
                # Effect size interpretation
                if abs(statistic) > 3:
                    insights.append(f"- **Large Effect**: Test statistic ({statistic:.2f}) suggests a substantial practical difference")
                elif abs(statistic) > 1.5:
                    insights.append(f"- **Medium Effect**: Test statistic ({statistic:.2f}) suggests a moderate practical difference")
                else:
                    insights.append(f"- **Small Effect**: Test statistic ({statistic:.2f}) suggests a small practical difference")
                
                # Practical significance vs statistical significance
                if p_val < 0.05 and abs(statistic) < 1:
                    insights.append("- **Caution**: While statistically significant, the effect size is small - consider practical significance")
                elif p_val >= 0.05 and abs(statistic) > 2:
                    insights.append("- **Note**: Large effect size despite non-significant p-value may indicate insufficient sample size")
                
                insights.append("")
        
        return "\n".join(insights)
    
    elif test_name == "ANOVA" and isinstance(result, dict):
        insights = []
        
        if 'f_statistic' in result and 'p_value' in result:
            f_stat = result['f_statistic']
            p_val = result['p_value']
            
            insights.append("**ANOVA Analysis:**")
            
            # F-statistic interpretation
            if f_stat > 10:
                insights.append(f"- Large F-statistic ({f_stat:.2f}) indicates substantial between-group differences")
            elif f_stat > 3:
                insights.append(f"- Moderate F-statistic ({f_stat:.2f}) suggests meaningful group differences")
            else:
                insights.append(f"- Small F-statistic ({f_stat:.2f}) indicates limited between-group variation")
            
            # P-value interpretation with ANOVA context
            if p_val < 0.05:
                insights.append(f"- Significant group differences detected (p = {p_val:.4f})")
                insights.append("- **Next Step**: Consider post-hoc tests to identify which specific groups differ")
            else:
                insights.append(f"- No significant group differences found (p = {p_val:.3f})")
            
            # Sample size considerations for ANOVA
            groups = data.select_dtypes(include=['object']).columns
            if len(groups) > 0:
                group_col = groups[0]
                group_sizes = data[group_col].value_counts()
                min_group_size = group_sizes.min()
                
                if min_group_size < 5:
                    insights.append(f"- **Critical Issue**: Smallest group has only {min_group_size} observations - results unreliable")
                elif min_group_size < 15:
                    insights.append(f"- **Power Limitation**: Smallest group has {min_group_size} observations - low power for detecting effects")
                else:
                    insights.append(f"- **Adequate Power**: Group sizes range from {min_group_size} to {group_sizes.max()} observations")
        
        return "\n".join(insights)
    
    elif test_name == "Correlation":
        insights = []
        insights.append("**Correlation Analysis:**")
        
        if isinstance(result, pd.DataFrame):
            # Find strongest correlations (excluding diagonal)
            corr_values = []
            for i in range(len(result.columns)):
                for j in range(i+1, len(result.columns)):
                    col1, col2 = result.columns[i], result.columns[j]
                    corr_val = result.iloc[i, j]
                    if pd.notna(corr_val):
                        corr_values.append((f"{col1} vs {col2}", corr_val))
            
            if corr_values:
                corr_values.sort(key=lambda x: abs(x[1]), reverse=True)
                strongest = corr_values[0]
                
                if abs(strongest[1]) > 0.8:
                    insights.append(f"- **Very Strong Correlation**: {strongest[0]} (r = {strongest[1]:.3f}) - potential multicollinearity concern")
                elif abs(strongest[1]) > 0.6:
                    insights.append(f"- **Strong Correlation**: {strongest[0]} (r = {strongest[1]:.3f}) - meaningful relationship")
                elif abs(strongest[1]) > 0.3:
                    insights.append(f"- **Moderate Correlation**: {strongest[0]} (r = {strongest[1]:.3f}) - weak to moderate relationship")
                else:
                    insights.append(f"- **Weak Correlations**: Strongest correlation is {strongest[0]} (r = {strongest[1]:.3f})")
                
                # Sample size impact on correlation
                if n_samples < 30:
                    insights.append(f"- **Reliability Concern**: With n={n_samples}, correlation estimates are unstable")
                elif n_samples < 100:
                    insights.append(f"- **Moderate Confidence**: n={n_samples} provides reasonable correlation estimates")
                else:
                    insights.append(f"- **High Confidence**: n={n_samples} provides stable correlation estimates")
        
        return "\n".join(insights)
    
    else:
        return f"**{test_name}**: Detailed interpretation not available for this analysis type."
        report.write("-" * 20 + "\n")
        report.write(f"Numeric columns: {len(data.select_dtypes(include=[np.number]).columns)}\n")
        report.write(f"Categorical columns: {len(data.select_dtypes(include=['object']).columns)}\n")
        report.write(f"Missing values: {data.isnull().sum().sum()}\n\n")
    
    if "Column Descriptions" in export_options:
        report.write("COLUMN DESCRIPTIONS:\n")
        report.write("-" * 20 + "\n")
        for col, desc in descriptions.items():
            if desc and desc.strip():
                report.write(f"{col}: {desc}\n")
        report.write("\n")
    
    if "Statistical Results" in export_options:
        report.write("STATISTICAL RESULTS:\n")
        report.write("-" * 20 + "\n")
        for test_name, result in results.items():
            report.write(f"\n{test_name}:\n")
            if isinstance(result, pd.DataFrame):
                report.write(result.to_string() + "\n")
            elif isinstance(result, list):
                for i, item in enumerate(result, 1):
                    report.write(f"  Test {i}: {item}\n")
            else:
                report.write(str(result) + "\n")
        report.write("\n")
    
    if "AI Insights" in export_options and hasattr(st.session_state, 'ai_insights_results'):
        insights = st.session_state.ai_insights_results
        report.write("AI INSIGHTS:\n")
        report.write("-" * 20 + "\n")
        report.write(f"Key Findings:\n{insights.get('key_findings', 'Not available')}\n\n")
        report.write(f"Recommendations:\n{insights.get('recommendations', 'Not available')}\n\n")
        report.write(f"Statistical Significance:\n{insights.get('significance', 'Not available')}\n\n")
    
    return report.getvalue()

if __name__ == "__main__":
    main()
