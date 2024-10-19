import unittest
from unittest.mock import patch, mock_open
import os
from disease_predict import load_models

class TestDiseasePredict(unittest.TestCase):

    @patch('disease_predict.os.listdir')
    @patch('disease_predict.open', new_callable=mock_open, read_data=b'some binary data')
    @patch('disease_predict.pickle.load')
    def test_load_models(self, mock_pickle_load, mock_open, mock_listdir):
        # Setup mock return values
        mock_listdir.return_value = ['model1.pkl', 'model2.pkl']
        mock_pickle_load.side_effect = ['model1', 'model2']

        # Call the function
        models, model_names = load_models()

        # Assertions
        self.assertEqual(models, ['model1', 'model2'])
        self.assertEqual(model_names, ['model1', 'model2'])

        # Verify that the correct files were opened
        mock_open.assert_any_call(os.path.join(os.getcwd(), 'models', 'model1.pkl'), 'rb')
        mock_open.assert_any_call(os.path.join(os.getcwd(), 'models', 'model2.pkl'), 'rb')

if __name__ == '__main__':
    unittest.main()