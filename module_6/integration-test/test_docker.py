# test_integration.py
import requests
import base64
import json
from deepdiff import DeepDiff


def test_lambda_integration():
    with open('event.json', 'rt', encoding='utf-8') as f_in:
        event = json.load(f_in)

    url = 'http://localhost:8180/2015-03-31/functions/function/invocations'
    response = requests.post(url, json=event)
    actual_response = response.json()
    print('actual_response:')
    print(json.dumps(actual_response, indent=2))

    expected_response = {
        'predictions': [{
            'model': 'ride_duration_prediction_model',
            'version': 123,
            'prediction': {
                'ride_duration': 18.230339990625332,
                'ride_id': 256
            }
        }]
    }

    diff = DeepDiff(actual_response, expected_response, ignore_order=True)
    print('Diff:', diff)
    #assert actual_response == expected_response
    #assert 'predictions' in actual_response
    #assert len(actual_response['predictions']) == 1
    #assert 'ride_duration' in actual_response['predictions'][0]['prediction']
    assert 'type_changes' not in diff
    assert 'values_changed' not in diff

    print("✅ Integration test passed!")
    print(f"Prediction: {actual_response['predictions'][0]['prediction']['ride_duration']}")


if __name__ == "__main__":
    test_lambda_integration()