import json
import os
import pandas as pd
import numpy as np
from openai import OpenAI

class AIInsights:
    def __init__(self):
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        self.model = "gpt-4o"
    
    def generate_insights(self, data, statistical_results, column_descriptions):
        """Generate AI-powered insights from statistical analysis"""
        try:
            # Prepare data summary
            data_summary = self._prepare_data_summary(data, statistical_results, column_descriptions)
            
            # Create prompt for AI analysis
            prompt = self._create_analysis_prompt(data_summary)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert statistician and data analyst. Provide clear, "
                        + "actionable insights from statistical analysis results. Focus on practical "
                        + "implications and recommendations. Always mention statistical significance "
                        + "and effect sizes when discussing results. Respond with JSON in the format: "
                        + "{'key_findings': 'text', 'recommendations': 'text', 'significance': 'text'}"
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            if content:
                insights = json.loads(content)
                return insights
            else:
                return {
                    "key_findings": "No content received from AI service",
                    "recommendations": "Please try again or check your API configuration",
                    "significance": "Unable to analyze due to empty response"
                }
            
        except Exception as e:
            return {
                "key_findings": f"Error generating insights: {str(e)}",
                "recommendations": "Please check your OpenAI API key and try again.",
                "significance": "Unable to analyze statistical significance due to error."
            }
    
    def _prepare_data_summary(self, data, statistical_results, column_descriptions):
        """Prepare a comprehensive data summary for AI analysis"""
        summary = {
            "dataset_info": {
                "shape": data.shape,
                "columns": list(data.columns),
                "numeric_columns": list(data.select_dtypes(include=[np.number]).columns),
                "categorical_columns": list(data.select_dtypes(include=['object']).columns),
                "missing_values": dict(data.isnull().sum())
            },
            "column_descriptions": column_descriptions,
            "statistical_results": {}
        }
        
        # Process statistical results
        for test_name, results in statistical_results.items():
            if test_name == "Descriptive Statistics" and isinstance(results, pd.DataFrame):
                summary["statistical_results"][test_name] = results.to_dict()
            elif test_name in ["T-Test", "ANOVA"] and isinstance(results, list):
                significant_results = [r for r in results if r.get('significant', False)]
                summary["statistical_results"][test_name] = {
                    "total_tests": len(results),
                    "significant_tests": len(significant_results),
                    "significant_results": significant_results[:5]  # Top 5 significant results
                }
            elif test_name == "Regression" and isinstance(results, dict):
                # Include only the most relevant regression results
                regression_summary = {}
                for reg_name, reg_result in list(results.items())[:3]:  # Top 3 regressions
                    if isinstance(reg_result, dict) and 'r_squared' in reg_result:
                        regression_summary[reg_name] = {
                            'r_squared': reg_result.get('r_squared', 0),
                            'f_pvalue': reg_result.get('f_pvalue', 1),
                            'significant_predictors': [
                                var for var, pval in reg_result.get('p_values', {}).items() 
                                if pval < 0.05
                            ]
                        }
                summary["statistical_results"][test_name] = regression_summary
            elif test_name == "Correlation" and isinstance(results, pd.DataFrame):
                # Find strong correlations
                strong_correlations = []
                for col1 in results.columns:
                    for col2 in results.columns:
                        if col1 != col2 and abs(results.loc[col1, col2]) > 0.5:
                            strong_correlations.append({
                                'variables': f"{col1} - {col2}",
                                'correlation': results.loc[col1, col2]
                            })
                
                summary["statistical_results"][test_name] = {
                    "strong_correlations": strong_correlations[:10]  # Top 10
                }
        
        return summary
    
    def _create_analysis_prompt(self, data_summary):
        """Create a detailed prompt for AI analysis"""
        prompt = f"""
        Please analyze the following statistical analysis results and provide insights:

        DATASET INFORMATION:
        - Shape: {data_summary['dataset_info']['shape']}
        - Numeric columns: {data_summary['dataset_info']['numeric_columns']}
        - Categorical columns: {data_summary['dataset_info']['categorical_columns']}
        - Missing values: {data_summary['dataset_info']['missing_values']}

        COLUMN DESCRIPTIONS:
        {self._format_column_descriptions(data_summary['column_descriptions'])}

        STATISTICAL RESULTS:
        {self._format_statistical_results(data_summary['statistical_results'])}

        Please provide:
        1. Key findings from the statistical analysis
        2. Practical recommendations based on the results
        3. Discussion of statistical significance and what it means

        Consider the experimental context provided in the column descriptions when interpreting results.
        Focus on practical implications and actionable insights.
        """
        
        return prompt
    
    def _format_column_descriptions(self, descriptions):
        """Format column descriptions for the prompt"""
        formatted = ""
        for col, desc in descriptions.items():
            if desc and desc.strip():
                formatted += f"- {col}: {desc}\n"
        return formatted if formatted else "No column descriptions provided."
    
    def _format_statistical_results(self, results):
        """Format statistical results for the prompt"""
        formatted = ""
        
        for test_name, test_results in results.items():
            formatted += f"\n{test_name}:\n"
            
            if test_name == "Descriptive Statistics":
                formatted += "- Basic descriptive statistics calculated for all numeric variables\n"
            
            elif test_name in ["T-Test", "ANOVA"]:
                total_tests = test_results.get('total_tests', 0)
                significant_tests = test_results.get('significant_tests', 0)
                formatted += f"- Total tests performed: {total_tests}\n"
                formatted += f"- Statistically significant results: {significant_tests}\n"
                
                if test_results.get('significant_results'):
                    formatted += "- Most significant findings:\n"
                    for result in test_results['significant_results'][:3]:
                        formatted += f"  * {result.get('test_name', 'Unknown')}: p-value = {result.get('p_value', 'N/A'):.4f}\n"
            
            elif test_name == "Regression":
                formatted += "- Regression analysis results:\n"
                for reg_name, reg_data in test_results.items():
                    if isinstance(reg_data, dict):
                        r_squared = reg_data.get('r_squared', 0)
                        f_pvalue = reg_data.get('f_pvalue', 1)
                        formatted += f"  * {reg_name}: R² = {r_squared:.4f}, F p-value = {f_pvalue:.4f}\n"
            
            elif test_name == "Correlation":
                strong_corr = test_results.get('strong_correlations', [])
                formatted += f"- Strong correlations found: {len(strong_corr)}\n"
                for corr in strong_corr[:3]:
                    formatted += f"  * {corr['variables']}: r = {corr['correlation']:.4f}\n"
        
        return formatted if formatted else "No statistical results to analyze."
    
    def generate_methodology_suggestions(self, column_descriptions):
        """Generate suggestions for improving experimental methodology"""
        try:
            prompt = f"""
            Based on the following experimental setup descriptions, provide suggestions for improving 
            the methodology and statistical analysis approach:

            EXPERIMENTAL SETUP:
            {self._format_column_descriptions(column_descriptions)}

            Please provide suggestions for:
            1. Improving data collection methods
            2. Additional statistical tests that might be relevant
            3. Potential confounding variables to consider
            4. Sample size considerations

            Respond with JSON in the format:
            {{"data_collection": "suggestions", "statistical_tests": "suggestions", 
              "confounding_variables": "suggestions", "sample_size": "suggestions"}}
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert in experimental design and statistical methodology."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=800
            )
            
            content = response.choices[0].message.content
            if content:
                return json.loads(content)
            else:
                return {
                    "data_collection": "No content received from AI service",
                    "statistical_tests": "Please try again or check your API configuration",
                    "confounding_variables": "Unable to analyze due to empty response",
                    "sample_size": "Unable to provide suggestions due to error"
                }
            
        except Exception as e:
            return {
                "data_collection": f"Error generating suggestions: {str(e)}",
                "statistical_tests": "Please check your OpenAI API key and try again.",
                "confounding_variables": "Unable to analyze due to error.",
                "sample_size": "Unable to provide suggestions due to error."
            }
