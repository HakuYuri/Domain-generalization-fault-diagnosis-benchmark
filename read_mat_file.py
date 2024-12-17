import numpy as np
from scipy.io import loadmat

data_param = [
    {
        'class': 'original',
        'data': [
            {
                'description': 'healthy',
                'mat_file_path': './dataset/result_1/solution.mat',
                'position': ['results', 'IR', 'accY']
            },
            {
                'description': 'or_fault',
                'mat_file_path': './dataset/result_2/solution.mat',
                'position': ['results', 'IR', 'accY']
            },
            {
                'description': 'ir_fault',
                'mat_file_path': './dataset/result_3/solution.mat',
                'position': ['results', 'IR', 'accY']
            },
        ]
    },
    {
        'class': 'mass_changed',
        'data': [
            {
                'description': 'healthy',
                'mat_file_path': './dataset/result_4/solution.mat',
                'position': ['results', 'IR', 'accY']
            },
            {
                'description': 'or_fault',
                'mat_file_path': './dataset/result_5/solution.mat',
                'position': ['results', 'IR', 'accY']
            },
            {
                'description': 'ir_fault',
                'mat_file_path': './dataset/result_6/solution.mat',
                'position': ['results', 'IR', 'accY']
            },
        ]
    },
    {
        'class': 'stiffness_changed',
        'data': [
            {
                'description': 'healthy',
                'mat_file_path': './dataset/result_7/solution.mat',
                'position': ['results', 'IR', 'accY']
            },
            {
                'description': 'or_fault',
                'mat_file_path': './dataset/result_8/solution.mat',
                'position': ['results', 'IR', 'accY']
            },
            {
                'description': 'ir_fault',
                'mat_file_path': './dataset/result_9/solution.mat',
                'position': ['results', 'IR', 'accY']
            },
        ]
    }
]


# Function to safely extract nested variables
def extract_nested_variable(data_dict, position_list):
    current = data_dict
    for key in position_list:
        # If current is a numpy structured array or dictionary
        if isinstance(current, np.ndarray) and current.dtype.names:
            current = current[0][key][0]
        # If current is a dictionary
        elif isinstance(current, dict):
            current = current[key]
        # If current is a numpy array, try to access first element if needed
        elif isinstance(current, np.ndarray):
            current = current[0][key]
        else:
            raise ValueError(f"Cannot extract {key} from current data type: {type(current)}")
    return current


def read_mat(params):
    data_dict = {}
    for single_class in data_param:

        for data_info in single_class['data']:
            # Load the .mat file
            path = data_info['mat_file_path']
            data = loadmat(data_info['mat_file_path'])
            # Extract the variable using the position
            position = data_info['position']

            print(f'path: {path}, position: {position}')
            try:
                extracted_data = extract_nested_variable(data, position)
                print(f"Extracted data type: {type(extracted_data)}")
                print(f"Extracted data shape: {extracted_data.shape if hasattr(extracted_data, 'shape') else 'N/A'}")

                data_dict[f"{single_class['class']}_{data_info['description']}"] = extracted_data

            except Exception as e:
                print(f"Error extracting data: {e}")
    return data_dict



