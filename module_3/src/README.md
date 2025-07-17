# Source Code

This directory contains the Python source code for the NYC Taxi Duration Prediction workflow:

- `model_utils.py` - Core utility functions for data preparation, model training, and MLflow integration
- `taxi_prediction_flow.py` - Main Prefect workflow implementation for orchestrating the ML pipeline
- `test_workflow.py` - End-to-end testing utilities for validating the workflow

## Main Components

### Model Utilities

The `model_utils.py` file contains functions for:
- Reading and preprocessing NYC taxi data
- Feature engineering for the prediction task
- Training the linear regression model
- MLflow integration for experiment tracking
- Model persistence and loading

### Prefect Workflow

The `taxi_prediction_flow.py` implements:
- Task definitions for each step in the ML pipeline
- Data validation and error handling
- Scheduling configuration for periodic training
- Parameter management for the workflow
- MLflow logging integration

### Testing Utilities

The `test_workflow.py` provides:
- End-to-end testing of the workflow
- Validation of MLflow experiment logging
- Environment setup validation
- Data access tests

## Usage

These modules are typically imported from the project root:

```python
from src.model_utils import read_dataframe, train_linear_model
from src.taxi_prediction_flow import taxi_duration_prediction_workflow
```

For more information, see the main project README.md file.
