import torch
import time
from thop import profile
import pandas as pd
from tabulate import tabulate
from libs.models.get_model import get_model_by_name


def compute_metrics_for_model(model_name, num_classes, in_width=512):
    # Get the model based on the provided name
    model = get_model_by_name(model_name, num_classes, in_width)

    # Input to the model
    input_data_1 = torch.randn(1, 3, in_width, in_width)
    input_data_2 = torch.randn(1, 3, in_width, in_width)

    # Move the model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        input_data_1 = input_data_1.cuda()
        input_data_2 = input_data_2.cuda()

    # Use thop's profile function to compute FLOPs and parameters
    flops, params = profile(model, inputs=(input_data_1, input_data_2))
    madds = flops / 2 / 1e9  # Convert FLOPs to MAdds

    # Convert params to millions
    params_in_millions = params / 1e6

    # Measure inference time to calculate FPS
    num_samples = 100  # Adjust the number of samples as needed
    start_time = time.time()
    for _ in range(num_samples):
        with torch.no_grad():
            _ = model(input_data_1, input_data_2)
    inference_time = time.time() - start_time
    fps = num_samples / inference_time

    # Round the values to two decimal places
    madds_rounded = round(madds, 2)
    params_in_millions_rounded = round(params_in_millions, 2)
    fps_rounded = round(fps, 2)

    # Create a dictionary for the model metrics
    model_metrics = {
        "Model": model_name,
        "Total MAdds": madds_rounded,
        "Total Parameters": params_in_millions_rounded,
        "FPS": fps_rounded
    }

    return model_metrics


# List of models to compute metrics for
model_names = ['SSCDL', 'BiSRNet', 'TED', 'SCanNet', 'A2Net', 'A2Net18']

# Specify common parameters (num_classes, in_width, etc.)
num_classes = 7
in_width = 512

# Collect metrics for each model
all_model_metrics = []
for model_name in model_names:
    model_metrics = compute_metrics_for_model(model_name, num_classes, in_width)
    all_model_metrics.append(model_metrics)

# Create a DataFrame from the list of dictionaries
result_df = pd.DataFrame(all_model_metrics)

# Display the table using tabulate
table_str = tabulate(result_df, headers='keys', tablefmt='pretty', showindex=False)

# Print the table
print(table_str)
