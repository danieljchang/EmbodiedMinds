# Visual In-Context Learning with 3D Perception

This project implements a model for visual in-context learning with 3D perception. The architecture combines various components such as depth estimation, object detection, and a fusion module to process and understand 3D environments based on visual and textual inputs.

## Project Structure

- **data/**: Contains datasets used in the project.
  - **raw/**: Raw data files.
  - **processed/**: Processed data files after transformations.
  - **tests/**: Test datasets for validation.

- **src/**: Source code for the project.
  - **datasets/**: Data loading and transformation utilities.
  - **preprocessing/**: Models for depth estimation and object detection.
  - **encoders/**: Classes for encoding visual and textual data.
  - **fusion/**: Module for combining 2D and 3D data.
  - **policy/**: Reasoning engine for mapping instructions to actions.
  - **heads/**: Output heads for action classification.
  - **training/**: Training and evaluation scripts.
  - **inference/**: Inference logic for predictions.
  - **utils/**: Utility functions for I/O and metrics.

- **configs/**: Configuration files for model parameters and training settings.

- **notebooks/**: Jupyter notebooks for experiments and visualizations.

- **scripts/**: Shell scripts for preprocessing and training.

- **tests/**: Unit tests for data and model functionalities.

## Installation

To set up the environment, you can use the provided `requirements.txt` or `environment.yml` file. 

### Using requirements.txt

```bash
pip install -r requirements.txt
```

### Using environment.yml

```bash
conda env create -f environment.yml
```

## Usage

1. **Preprocess the Data**: Run the preprocessing script to prepare the raw data.
   ```bash
   bash scripts/preprocess.sh
   ```

2. **Train the Model**: Start the training process using the training script.
   ```bash
   bash scripts/run_train.sh
   ```

3. **Inference**: Use the inference script to make predictions with the trained model.

## Testing

Unit tests are provided to ensure the functionality of data loading and model components. You can run the tests using:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.