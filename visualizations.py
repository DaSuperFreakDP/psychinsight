import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns

class Visualizations:
    def __init__(self):
        # Set default color palette
        self.color_palette = px.colors.qualitative.Set3
    
    def create_histogram(self, data, column, bins=30):
        """Create an interactive histogram"""
        try:
            fig = px.histogram(
                data, 
                x=column,
                nbins=bins,
                title=f'Distribution of {column}',
                labels={column: column, 'count': 'Frequency'},
                color_discrete_sequence=self.color_palette
            )
            
            # Add mean line
            mean_val = data[column].mean()
            fig.add_vline(
                x=mean_val, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Mean: {mean_val:.2f}"
            )
            
            # Add median line
            median_val = data[column].median()
            fig.add_vline(
                x=median_val, 
                line_dash="dash", 
                line_color="orange",
                annotation_text=f"Median: {median_val:.2f}"
            )
            
            fig.update_layout(
                showlegend=True,
                height=500,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            # Return empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating histogram: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_scatter_plot(self, data, x_col, y_col, color_col=None):
        """Create an interactive scatter plot"""
        try:
            fig = px.scatter(
                data, 
                x=x_col, 
                y=y_col,
                color=color_col if color_col and color_col in data.columns else None,
                title=f'{y_col} vs {x_col}',
                labels={x_col: x_col, y_col: y_col},
                color_discrete_sequence=self.color_palette,
                hover_data=data.columns.tolist()
            )
            
            # Add trendline
            if data[x_col].notna().sum() > 1 and data[y_col].notna().sum() > 1:
                z = np.polyfit(data[x_col].dropna(), data[y_col].dropna(), 1)
                p = np.poly1d(z)
                
                x_trend = np.linspace(data[x_col].min(), data[x_col].max(), 100)
                y_trend = p(x_trend)
                
                fig.add_traces(go.Scatter(
                    x=x_trend, 
                    y=y_trend,
                    mode='lines',
                    name='Trendline',
                    line=dict(color='red', dash='dash')
                ))
                
                # Calculate and display correlation
                corr = data[x_col].corr(data[y_col])
                fig.add_annotation(
                    text=f"Correlation: {corr:.3f}",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98, showarrow=False,
                    bgcolor="white", bordercolor="black", borderwidth=1
                )
            
            fig.update_layout(height=500)
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating scatter plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_box_plot(self, data, numeric_col, categorical_col=None):
        """Create an interactive box plot"""
        try:
            if categorical_col and categorical_col in data.columns:
                fig = px.box(
                    data, 
                    x=categorical_col, 
                    y=numeric_col,
                    title=f'Distribution of {numeric_col} by {categorical_col}',
                    color=categorical_col,
                    color_discrete_sequence=self.color_palette
                )
            else:
                fig = px.box(
                    data, 
                    y=numeric_col,
                    title=f'Distribution of {numeric_col}',
                    color_discrete_sequence=self.color_palette
                )
            
            # Add mean markers
            if categorical_col and categorical_col in data.columns:
                means = data.groupby(categorical_col)[numeric_col].mean()
                for cat, mean_val in means.items():
                    fig.add_hline(
                        y=mean_val,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Mean: {mean_val:.2f}"
                    )
            else:
                mean_val = data[numeric_col].mean()
                fig.add_hline(
                    y=mean_val,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Overall Mean: {mean_val:.2f}"
                )
            
            fig.update_layout(height=500)
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating box plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_correlation_heatmap(self, data):
        """Create a correlation heatmap"""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) < 2:
                fig = go.Figure()
                fig.add_annotation(
                    text="Need at least 2 numeric columns for correlation heatmap",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            corr_matrix = numeric_data.corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Correlation Matrix",
                color_continuous_scale="RdBu_r",
                aspect="auto",
                text_auto=True
            )
            
            fig.update_layout(
                height=600,
                xaxis_title="Variables",
                yaxis_title="Variables"
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating correlation heatmap: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_statistical_summary_plot(self, statistical_results):
        """Create visualizations for statistical test results"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("P-Values Distribution", "Effect Sizes", 
                               "Test Statistics", "Significance Summary"),
                specs=[[{"type": "histogram"}, {"type": "bar"}],
                       [{"type": "histogram"}, {"type": "pie"}]]
            )
            
            # Collect data from statistical results
            p_values = []
            effect_sizes = []
            test_statistics = []
            significant_tests = 0
            total_tests = 0
            
            for test_name, results in statistical_results.items():
                if test_name in ["T-Test", "ANOVA"] and isinstance(results, list):
                    for result in results:
                        if 'p_value' in result:
                            p_values.append(result['p_value'])
                            total_tests += 1
                            if result.get('significant', False):
                                significant_tests += 1
                        if 'effect_size' in result:
                            effect_sizes.append(result['effect_size'])
                        if 'statistic' in result:
                            test_statistics.append(abs(result['statistic']))
            
            # P-values histogram
            if p_values:
                fig.add_trace(
                    go.Histogram(x=p_values, nbinsx=20, name="P-Values"),
                    row=1, col=1
                )
            
            # Effect sizes bar chart
            if effect_sizes:
                fig.add_trace(
                    go.Bar(x=list(range(len(effect_sizes))), y=effect_sizes, name="Effect Sizes"),
                    row=1, col=2
                )
            
            # Test statistics histogram
            if test_statistics:
                fig.add_trace(
                    go.Histogram(x=test_statistics, nbinsx=15, name="Test Statistics"),
                    row=2, col=1
                )
            
            # Significance pie chart
            if total_tests > 0:
                fig.add_trace(
                    go.Pie(
                        labels=["Significant", "Not Significant"],
                        values=[significant_tests, total_tests - significant_tests],
                        name="Significance"
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=800,
                title_text="Statistical Tests Summary",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating statistical summary plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_regression_plot(self, data, regression_results):
        """Create regression diagnostic plots"""
        try:
            if not regression_results:
                fig = go.Figure()
                fig.add_annotation(
                    text="No regression results available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            # Get the first regression result for plotting
            reg_name = list(regression_results.keys())[0]
            reg_data = regression_results[reg_name]
            
            if 'Multiple Regression' in regression_results:
                reg_name = 'Multiple Regression'
                reg_data = regression_results[reg_name]
            
            # Extract dependent and independent variables from regression name
            if ' ~ ' in reg_name:
                dependent_var, independent_var = reg_name.split(' ~ ')
                
                # Create scatter plot with regression line
                fig = self.create_scatter_plot(data, independent_var, dependent_var)
                
                # Add R-squared annotation
                r_squared = reg_data.get('r_squared', 0)
                fig.add_annotation(
                    text=f"R² = {r_squared:.3f}",
                    xref="paper", yref="paper",
                    x=0.02, y=0.85, showarrow=False,
                    bgcolor="white", bordercolor="black", borderwidth=1
                )
            
            else:
                # For multiple regression, create a general info plot
                fig = go.Figure()
                
                text = f"Multiple Regression Analysis<br>"
                text += f"R² = {reg_data.get('r_squared', 0):.4f}<br>"
                text += f"Adjusted R² = {reg_data.get('adj_r_squared', 0):.4f}<br>"
                text += f"F p-value = {reg_data.get('f_pvalue', 1):.4f}<br>"
                text += f"Observations = {reg_data.get('observations', 0)}"
                
                fig.add_annotation(
                    text=text,
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16)
                )
            
            fig.update_layout(title=f"Regression Analysis: {reg_name}")
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating regression plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
