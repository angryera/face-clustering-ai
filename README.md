# FACE CLUSTERING AI

This project is designed to implement an AI model using Python. It includes functionalities for loading datasets, training models, and making predictions.

## Purpose

The purpose of this project is to develop a face recognition system using advanced machine learning techniques. The system aims to identify and verify individuals by analyzing facial features, making it suitable for applications such as security, authentication, and personalized user experiences.

## Project Structure

```
face-clustering-ai
├── src
│   ├── main.py          # Entry point of the application
│   ├── models
│   │   └── model.py     # Contains the Model class for training and prediction
│   ├── data
│   │   └── dataset.py    # Contains the Dataset class for loading and preprocessing data
│   └── utils
│       └── helpers.py    # Contains utility functions for logging and saving models
├── requirements.txt      # Lists the project dependencies
├── README.md             # Documentation for the project
└── .gitignore            # Specifies files to be ignored by Git
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/my-ai-project.git
   cd my-ai-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. If `pip install dlib` fails, follow these steps to build `dlib` manually:
   - Install CMake, a C++ compiler, and Python development headers.
   - Clone the `dlib` repository:
     ```
     git clone https://github.com/davisking/dlib.git
     cd dlib
     ```
   - Build and install `dlib`:
     ```
     mkdir build
     cd build
     cmake ..
     cmake --build . --config Release
     cd ..
     python setup.py install
     ```

## Usage

To run the application, execute the following command:
```
python src/main.py
```

This will initialize the application, load the dataset, and train the AI model.

## Contributing

Feel free to submit issues or pull requests for improvements or bug fixes.