# Smartphone_Price_prediction

# Mini Project: Smartphone Price Prediction

## Overview

This project provides a machine learning solution for predicting smartphone prices (both online and retail) using specifications such as RAM and ROM. The implementation leverages Recurrent Neural Networks (RNNs) with TensorFlow and offers static and animated visualizations of the predictions.

---

## Project Structure

```
Mini_Project/
├── main1.py                 # Script for RNN-based price prediction with static visualization
├── main_3d.py               # Script with RNN-based animation price prediction and animated visualization
├── PhoneModel1.csv          # Dataset containing smartphone specifications and prices
├── output_animation_style.webm # Animation video showcasing model predictions
├── output.png               # Static visualization of predictions
├── Final Docment.odt        # Project report or documentation
```

---

## Dataset

- **File:** `PhoneModel1.csv`
- **Description:** A dataset containing 162 entries of smartphone specifications and corresponding prices in both online and retail markets.
  - **Features:**
    - `Ram` (in GB)
    - `ROM` (in GB)
  - **Targets:**
    - `Online-Price` (in local currency)
    - `Retail-Price` (in local currency)

Example Rows:

| Brand                 | Ram | ROM | Online-Price | Retail-Price |
| --------------------- | --- | --- | ------------ | ------------ |
| Samsung galaxy A72 5G | 4   | 128 | 14999        | 12999        |
| Oppo K1               | 4   | 128 | 27499        | 23499        |

---

## Features

1. **Machine Learning Model:**
   - Implements an RNN with LSTM layers to model non-linear relationships between specifications and prices.
   - Data is standardized using `StandardScaler` for better model performance.
2. **Visualization Options:**
   - Static plots compare true and predicted prices.
   - An animated graph visualizes the prediction process over time.
3. **Libraries and Tools:**
   - TensorFlow and Keras for machine learning.
   - Matplotlib for visualizations.
   - Pandas and Scikit-learn for data processing.

---

## Workflow

### Data Processing

- Data is loaded from `PhoneModel1.csv`.
- Features (`Ram`, `ROM`) are standardized for uniform scaling.

### Model Training

- An LSTM-based RNN is trained on the processed data.
- Loss minimization and validation metrics are logged during training.

### Visualization

- Predictions are visualized in static plots (`output.png`) and animated graphs (`output_animation_style.webm`).

---

## Usage

### Prerequisites

- Python 3.8+ is required.
- Install the required dependencies:
  ```bash
  pip install tensorflow matplotlib pandas scikit-learn seaborn
  ```

### Running the Project

1. **Static Prediction and Visualization:**

   ```bash
   python main1.py
   ```

   - Outputs a static comparison plot (`output.png`).

2. **Animated Visualization:**

   ```bash
   python main_3d.py
   ```

   - Generates a dynamic prediction animation (`output_animation_style.webm`).

---

## Outputs

1. **Static Plot:**
   - Displays true vs. predicted prices for evaluation.
2. **Animated Visualization:**
   - Showcases how predictions evolve during the model's learning process.

---

## Future Enhancements

- **Additional Features:** Include specifications like processor, battery capacity, and camera quality for better predictions.
- **Advanced Models:** Explore Transformer models for sequence learning.
- **UI Development:** Build an interactive user interface to simplify data input and visualization.

---

## Challenges and Limitations

- **Data Availability:** The dataset focuses only on RAM and ROM specifications, which might not fully capture price determinants.
- **Generalization:** Model accuracy could vary for unseen smartphone models with unique configurations.
- **Performance:** Training RNNs can be computationally expensive for large datasets.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

Special thanks to open-source libraries and frameworks that made this project possible. TensorFlow, Matplotlib, and Pandas were particularly instrumental in the implementation.

