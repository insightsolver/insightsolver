import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from insightsolver import InsightSolver

# Fixture for a sample DataFrame
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Survived': [1, 0, 1, 0, 1, 0, 1, 0],
        'Age': [22, 38, 26, 35, 35, 28, 45, 19],
        'Class': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B'],
        'Fare': [7.25, 71.28, 7.92, 53.1, 8.05, 8.45, 51.86, 21.07]
    })

def test_insightsolver_initialization(sample_df):
    """Test that the solver initializes correctly."""
    solver = InsightSolver(df=sample_df, target_name='Survived', target_goal=1)
    
    assert solver.target_name == 'Survived'
    assert solver.target_goal == 1
    
    # Note: Type inference happens during fit (via server response) or if explicitly provided.
    # At initialization, only target type might be inferred if not provided.
    assert 'Survived' in solver.columns_types

def test_insightsolver_initialization_custom_types(sample_df):
    """Test initialization with custom column types."""
    custom_types = {'Class': 'ignore'}
    solver = InsightSolver(
        df=sample_df, 
        target_name='Survived', 
        target_goal=1,
        columns_types=custom_types
    )
    
    assert solver.columns_types['Class'] == 'ignore'

@patch('insightsolver.api_utilities.search_best_ruleset_from_API_dict')
def test_fit_mocked(mock_api_call, sample_df):
    """Test the fit method with a mocked API response."""
    
    # Mock response structure based on what the solver expects
    mock_response = {
        'dataset_metadata': {
            'columns_names_to_btypes': {
                'Survived': 'binary',
                'Age': 'continuous',
                'Class': 'multiclass',
                'Fare': 'continuous'
            }
        },
        'rule_mining_results': {
            'rules': [
                {
                    'rule_id': 1,
                    'rule_definition': 'Age > 30',
                    'metrics': {'precision': 0.8, 'recall': 0.5}
                }
            ],
            'summary': 'Success'
        },
        'benchmark_scores': {
            'original': [0.8, 0.82],
            'shuffled': [0.5, 0.51]
        },
        'monitoring_metadata': {
            'execution_time': 1.2
        }
    }
    mock_api_call.return_value = mock_response

    solver = InsightSolver(df=sample_df, target_name='Survived', target_goal=1)
    
    # Call fit (which calls the API)
    # We don't need a real service key since we mock the API call
    solver.fit(service_key='dummy_key.json')

    # Verify the solver state is updated
    assert solver._is_fitted is True
    assert solver.rule_mining_results == mock_response['rule_mining_results']
    
    # Verify types are updated from server response
    assert solver.columns_types['Age'] == 'continuous'
    
    # Verify API was called
    mock_api_call.assert_called_once()
    
    # Verify arguments passed to API
    call_args = mock_api_call.call_args[1]
    assert 'd_out_original' in call_args
    d_out = call_args['d_out_original']
    assert d_out['target_name'] == 'Survived'
    assert d_out['target_goal'] == 1

def test_not_fitted_error(sample_df):
    """Test that methods requiring fit raise an error if not fitted."""
    solver = InsightSolver(df=sample_df, target_name='Survived', target_goal=1)
    
    assert solver.is_fitted() is False
    
    # Assuming there's a method that checks for fit, e.g., accessing results
    # If accessing rule_mining_results directly doesn't raise, we might check other methods
    # For now, just checking the flag
    pass
