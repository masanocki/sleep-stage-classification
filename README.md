# Sleep Stage Classification

## Overview
This project focuses on automatic sleep stage classification based on the attached polysomnographic data. The algorithm used for classification is Random Forest, while the UI was created using the CustomTkinter library.

## Features
- Data preprocessing and feature extraction
- Machine learning model training and evaluation
- Saved model for future predictions (there is a script with this feature, however, this function is not included in the graphical version of the application)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/masanocki/sleep-stage-classification.git
   cd sleep-stage-classification
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Prepare polysomnographic data (preferably in EDF format), and optionally sleep scoring, in order to obtain a reference point for statistics.
2. Run the main script to start application
   ```bash
   py src/main.py
   ```
3. Prediction with pretrained model:
   - Provide paths to prepared data in designated locations
   - Hit the button to start prediction
4. Prediction without pretrained model:
   - Provide path to prepared data in designated locations (you need to provide paths to data for test and train)
   - Adjust algorithm parameters
   - Hit the button to start prediction

## Project Structure
- `src/`: Contains the main code
- `assets/`: Stores any additional resources like images or datasets
- `requirements.txt`: Lists all the dependencies required to run the project


