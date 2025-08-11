# Overview

This is a Statistical Analysis with AI Insights application built with Streamlit. The application allows users to upload experimental data (CSV or Excel files), perform comprehensive statistical analysis, and receive AI-powered insights using OpenAI's GPT-4o model. The system provides descriptive statistics, hypothesis testing, and interactive visualizations to help users understand their data and make data-driven decisions.

# User Preferences

Preferred communication style: Simple, everyday language.

## Recent Development Sessions
- Enhanced filtering: Column-by-column filtering with checkboxes for categorical data
- AI cell analysis: Context-aware insights based on column descriptions  
- Selection calculations: Row range selection for statistical calculations
- Column setup: Display names + hidden descriptions for AI analysis
- Performance assessment: Smart interpretation of "lower/higher is better" scenarios

## User-Requested Features Completed
- ✅ Column filtering with value counts (e.g., "Control (10)", "Treatment (11)")
- ✅ AI insight on individual cell values with context understanding
- ✅ Selection-based calculations for data ranges
- ✅ Enhanced column descriptions for better AI analysis
- ✅ Clear statistical analysis workflow with prominent buttons

## Project Status
All requested features implemented and functional. Application provides comprehensive statistical analysis with AI-powered insights, advanced filtering, and cell-level analysis capabilities.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit web application framework
- **Layout**: Wide layout with expandable sidebar for controls
- **Session State Management**: Uses Streamlit's session state to persist application components across interactions
- **File Upload**: Supports CSV and Excel file formats with multiple encoding fallbacks

## Backend Architecture
- **Modular Design**: Separated into distinct classes for specific functionalities:
  - `DataManager`: Handles data loading, cleaning, and column descriptions
  - `StatisticalAnalyzer`: Performs statistical computations and hypothesis testing
  - `AIInsights`: Integrates with OpenAI API for generating insights
  - `Visualizations`: Creates interactive plots using Plotly
- **Error Handling**: Comprehensive exception handling across all modules
- **Data Processing Pipeline**: Load → Clean → Analyze → Visualize → Generate AI Insights

## Statistical Analysis Components
- **Descriptive Statistics**: Mean, median, mode, variance, IQR, skewness, kurtosis
- **Hypothesis Testing**: T-tests and other statistical tests using SciPy
- **ANOVA**: Analysis of variance using statsmodels
- **Data Cleaning**: Automatic removal of empty rows/columns and whitespace trimming

## Visualization System
- **Interactive Charts**: Plotly-based histograms, scatter plots, and statistical visualizations
- **Color Consistency**: Unified color palette across all visualizations
- **Statistical Overlays**: Mean and median lines on distributions
- **Responsive Design**: Charts adapt to different screen sizes

# External Dependencies

## AI Services
- **OpenAI API**: GPT-4o model for generating statistical insights and recommendations
- **Authentication**: API key-based authentication via environment variables
- **Response Format**: Structured JSON responses with key findings, recommendations, and significance analysis

## Data Science Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Statistical functions and hypothesis testing
- **statsmodels**: Advanced statistical modeling and ANOVA

## Visualization Libraries
- **plotly**: Interactive plotting and charting
- **seaborn**: Statistical visualization support

## Web Framework
- **streamlit**: Web application framework for data science applications
- **File Processing**: Built-in support for CSV and Excel file uploads