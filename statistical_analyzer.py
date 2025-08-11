import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import warnings

warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    def __init__(self):
        pass
    
    def descriptive_stats(self, data):
        """Calculate descriptive statistics for all numeric columns"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return pd.DataFrame()
        
        # Basic descriptive statistics
        desc_stats = numeric_data.describe()
        
        # Add additional statistics
        additional_stats = pd.DataFrame(index=numeric_data.columns)
        additional_stats['variance'] = numeric_data.var()
        additional_stats['iqr'] = numeric_data.quantile(0.75) - numeric_data.quantile(0.25)
        additional_stats['skewness'] = numeric_data.skew()
        additional_stats['kurtosis'] = numeric_data.kurtosis()
        
        # Mode calculation (handling multiple modes)
        mode_values = []
        for col in numeric_data.columns:
            mode_result = stats.mode(numeric_data[col].dropna(), keepdims=False)
            if hasattr(mode_result, 'mode'):
                mode_values.append(mode_result.mode)
            else:
                mode_values.append(mode_result[0] if len(mode_result) > 0 else np.nan)
        
        additional_stats['mode'] = mode_values
        
        # Combine all statistics
        combined_stats = pd.concat([desc_stats.T, additional_stats], axis=1)
        
        return combined_stats.round(4)
    
    def t_test(self, data, alpha=0.05):
        """Perform various t-tests on the data"""
        results = []
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 1:
            return results
        
        # One-sample t-test for each numeric column (testing against 0)
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) > 1:
                t_stat, p_val = stats.ttest_1samp(col_data, 0)
                results.append({
                    'test_name': f'One-sample t-test: {col} vs 0',
                    'statistic': t_stat,
                    'p_value': p_val,
                    'significant': p_val < alpha,
                    'effect_size': abs(col_data.mean()) / col_data.std() if col_data.std() > 0 else 0
                })
        
        # Two-sample t-tests between numeric columns
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                data1 = data[col1].dropna()
                data2 = data[col2].dropna()
                
                if len(data1) > 1 and len(data2) > 1:
                    # Independent samples t-test
                    t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
                    
                    # Calculate Cohen's d
                    pooled_std = np.sqrt(((len(data1) - 1) * data1.var() + (len(data2) - 1) * data2.var()) / (len(data1) + len(data2) - 2))
                    cohens_d = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0
                    
                    results.append({
                        'test_name': f'Independent t-test: {col1} vs {col2}',
                        'statistic': t_stat,
                        'p_value': p_val,
                        'significant': p_val < alpha,
                        'effect_size': abs(cohens_d)
                    })
        
        # If there are categorical columns, perform t-tests between groups
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        for cat_col in categorical_cols:
            unique_categories = data[cat_col].dropna().unique()
            if len(unique_categories) == 2:  # Only for binary categorical variables
                for num_col in numeric_cols:
                    group1_data = data[data[cat_col] == unique_categories[0]][num_col].dropna()
                    group2_data = data[data[cat_col] == unique_categories[1]][num_col].dropna()
                    
                    if len(group1_data) > 1 and len(group2_data) > 1:
                        t_stat, p_val = stats.ttest_ind(group1_data, group2_data, equal_var=False)
                        
                        # Calculate Cohen's d
                        pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + (len(group2_data) - 1) * group2_data.var()) / (len(group1_data) + len(group2_data) - 2))
                        cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std if pooled_std > 0 else 0
                        
                        results.append({
                            'test_name': f'Group comparison: {num_col} by {cat_col}',
                            'statistic': t_stat,
                            'p_value': p_val,
                            'significant': p_val < alpha,
                            'effect_size': abs(cohens_d),
                            'group1_mean': group1_data.mean(),
                            'group2_mean': group2_data.mean(),
                            'group1': unique_categories[0],
                            'group2': unique_categories[1]
                        })
        
        return results
    
    def anova_test(self, data, alpha=0.05):
        """Perform ANOVA tests"""
        results = []
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        # One-way ANOVA for each numeric variable grouped by categorical variables
        for cat_col in categorical_cols:
            unique_categories = data[cat_col].dropna().unique()
            if len(unique_categories) >= 2:  # Need at least 2 groups
                for num_col in numeric_cols:
                    groups = [data[data[cat_col] == cat][num_col].dropna() for cat in unique_categories]
                    groups = [g for g in groups if len(g) > 0]  # Remove empty groups
                    
                    if len(groups) >= 2 and all(len(g) > 1 for g in groups):
                        try:
                            f_stat, p_val = stats.f_oneway(*groups)
                            
                            # Calculate eta-squared (effect size)
                            total_mean = data[num_col].mean()
                            ss_between = sum(len(g) * (g.mean() - total_mean)**2 for g in groups)
                            ss_total = sum((data[num_col] - total_mean)**2)
                            eta_squared = ss_between / ss_total if ss_total > 0 else 0
                            
                            results.append({
                                'test_name': f'One-way ANOVA: {num_col} by {cat_col}',
                                'statistic': f_stat,
                                'p_value': p_val,
                                'significant': p_val < alpha,
                                'effect_size': eta_squared,
                                'groups': len(groups),
                                'group_means': {cat: data[data[cat_col] == cat][num_col].mean() for cat in unique_categories}
                            })
                        except Exception as e:
                            continue
        
        return results
    
    def regression_analysis(self, data):
        """Perform regression analysis"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {}
        
        results = {}
        
        # Simple linear regression for all pairs
        for i, dependent_var in enumerate(numeric_cols):
            for independent_var in numeric_cols:
                if dependent_var != independent_var:
                    try:
                        # Prepare data
                        reg_data = data[[dependent_var, independent_var]].dropna()
                        
                        if len(reg_data) < 3:  # Need at least 3 points for regression
                            continue
                        
                        y = reg_data[dependent_var]
                        X = reg_data[independent_var]
                        X = sm.add_constant(X)  # Add intercept
                        
                        # Fit model
                        model = sm.OLS(y, X).fit()
                        
                        # Calculate additional metrics
                        y_pred = model.predict(X)
                        mse = np.mean((y - y_pred)**2)
                        rmse = np.sqrt(mse)
                        
                        regression_key = f"{dependent_var} ~ {independent_var}"
                        results[regression_key] = {
                            'r_squared': model.rsquared,
                            'adj_r_squared': model.rsquared_adj,
                            'f_statistic': model.fvalue,
                            'f_pvalue': model.f_pvalue,
                            'coefficients': dict(zip(model.params.index, model.params.values)),
                            'p_values': dict(zip(model.pvalues.index, model.pvalues.values)),
                            'mse': mse,
                            'rmse': rmse,
                            'observations': len(reg_data)
                        }
                        
                    except Exception as e:
                        continue
        
        # Multiple regression (using first numeric column as dependent variable)
        if len(numeric_cols) > 2:
            try:
                dependent_var = numeric_cols[0]
                independent_vars = numeric_cols[1:]
                
                # Prepare data
                formula_vars = [dependent_var] + list(independent_vars)
                reg_data = data[formula_vars].dropna()
                
                if len(reg_data) > len(independent_vars) + 1:  # Need more observations than predictors
                    y = reg_data[dependent_var]
                    X = reg_data[independent_vars]
                    X = sm.add_constant(X)
                    
                    # Fit model
                    model = sm.OLS(y, X).fit()
                    
                    results['Multiple Regression'] = {
                        'dependent_variable': dependent_var,
                        'independent_variables': list(independent_vars),
                        'r_squared': model.rsquared,
                        'adj_r_squared': model.rsquared_adj,
                        'f_statistic': model.fvalue,
                        'f_pvalue': model.f_pvalue,
                        'coefficients': dict(zip(model.params.index, model.params.values)),
                        'p_values': dict(zip(model.pvalues.index, model.pvalues.values)),
                        'observations': len(reg_data)
                    }
            except Exception as e:
                pass
        
        return results
    
    def correlation_analysis(self, data):
        """Perform correlation analysis"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return pd.DataFrame()
        
        # Pearson correlation
        corr_matrix = numeric_data.corr()
        
        # P-values for correlations
        p_values = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
        
        for col1 in numeric_data.columns:
            for col2 in numeric_data.columns:
                if col1 != col2:
                    data1 = numeric_data[col1].dropna()
                    data2 = numeric_data[col2].dropna()
                    
                    # Find common indices
                    common_idx = data1.index.intersection(data2.index)
                    if len(common_idx) > 2:
                        corr, p_val = stats.pearsonr(data1.loc[common_idx], data2.loc[common_idx])
                        p_values.loc[col1, col2] = p_val
                    else:
                        p_values.loc[col1, col2] = 1.0
                else:
                    p_values.loc[col1, col2] = 0.0
        
        # Create a combined result
        result_dict = {}
        for col1 in corr_matrix.index:
            for col2 in corr_matrix.columns:
                if col1 != col2:
                    key = f"{col1} - {col2}"
                    result_dict[key] = {
                        'correlation': corr_matrix.loc[col1, col2],
                        'p_value': p_values.loc[col1, col2],
                        'significant': p_values.loc[col1, col2] < 0.05
                    }
        
        return corr_matrix.round(4)
    
    def chi_square_test(self, data, col1, col2, alpha=0.05):
        """Perform chi-square test for independence between two categorical variables"""
        try:
            # Create contingency table
            contingency_table = pd.crosstab(data[col1], data[col2])
            
            # Perform chi-square test
            chi2, p_val, dof, expected = chi2_contingency(contingency_table)
            
            # Calculate Cramer's V (effect size)
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            
            return {
                'test_name': f'Chi-square test: {col1} vs {col2}',
                'chi2_statistic': chi2,
                'p_value': p_val,
                'degrees_of_freedom': dof,
                'significant': p_val < alpha,
                'effect_size': cramers_v,
                'contingency_table': contingency_table
            }
        except Exception as e:
            return None
    
    def normality_test(self, data, column):
        """Test for normality using Shapiro-Wilk test"""
        try:
            col_data = data[column].dropna()
            
            if len(col_data) < 3:
                return None
            
            # Shapiro-Wilk test (good for small samples)
            if len(col_data) <= 5000:
                stat, p_val = stats.shapiro(col_data)
                test_name = "Shapiro-Wilk"
            else:
                # Use Kolmogorov-Smirnov test for larger samples
                stat, p_val = stats.kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
                test_name = "Kolmogorov-Smirnov"
            
            return {
                'test_name': f'{test_name} normality test: {column}',
                'statistic': stat,
                'p_value': p_val,
                'normal_distribution': p_val > 0.05,
                'sample_size': len(col_data)
            }
        except Exception as e:
            return None
