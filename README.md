<div align="center">

# Deep Learning with Python
### Introduction to Neural Networks & Machine Learning

[![Stars](https://img.shields.io/github/stars/ADolbyB/deep-learning-python?style=for-the-badge&logo=github)](https://github.com/ADolbyB/deep-learning-python/stargazers)
[![Forks](https://img.shields.io/github/forks/ADolbyB/deep-learning-python?style=for-the-badge&logo=github)](https://github.com/ADolbyB/deep-learning-python/network/members)
[![Repo Size](https://img.shields.io/github/repo-size/ADolbyB/deep-learning-python?label=Repo%20Size&logo=Github&style=for-the-badge)](https://github.com/ADolbyB/deep-learning-python)
[![Last Commit](https://img.shields.io/github/last-commit/ADolbyB/deep-learning-python?style=for-the-badge&logo=github)](https://github.com/ADolbyB/deep-learning-python/commits/main)

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

**Developed by:** [![ADolbyB](https://img.shields.io/badge/ADolbyB-Profile-blue?style=for-the-badge&logo=github)](https://github.com/ADolbyB)

</div>

---

## 📚 Course Overview

**Course:** Introduction to Deep Learning
**Focus:** Practical implementation of neural networks and deep learning algorithms using Python

This repository contains a comprehensive collection of Jupyter notebooks, assignments, and practice implementations covering fundamental to advanced deep learning concepts. All code is written in Python using industry-standard frameworks including TensorFlow and Keras.

---

## 🎯 Learning Objectives

This repository demonstrates mastery of:

✅ **Neural Network Fundamentals** - Perceptrons, activation functions, backpropagation  
✅ **Deep Learning Architectures** - CNNs, RNNs, and specialized networks  
✅ **Gradient Descent Optimization** - SGD, Adam, RMSprop, learning rate scheduling  
✅ **TensorFlow & Keras** - Model building, training, and deployment  
✅ **Computer Vision** - Image classification, feature extraction, transfer learning  
✅ **Model Evaluation** - Training/validation splits, performance metrics, overfitting prevention  

---

## 📂 Repository Structure

```
deep-learning-python/
├── assets/                         # Handwritten solutions for assignments and lectures
│   ├── HW1/                        # Assets for assignment1.ipynb
│   ├── HW2/                        # Assets for assignment2.ipynb
│   ├── ...
│   ├── HW6/                        # Assets for assignment6.ipynb
│   ├── Lecture2/                   # Assets for lecture2.ipynb
│   └── Lecture6/                   # Assets for lecture6a/b/c/d.ipynb
├── Assignments/                    # Python code for Deep Learning
│   ├── assignment1-test.ipynb      # Test script for Assignment 1
│   ├── assignment1.ipynb           # Code for Assignment 1
│   ├── assignment2-test.ipynb      # Test script for Assignment 2
│   ├── assignment2.ipynb           # Code for Assignment 2
│   ├── ...
│   └── assignment6.ipynb           # Code for Assignment 6
├── Lectures/                       # Lecture notebooks and examples
│   ├── lecture1.ipynb              # Code from 1st week of lectures
│   ├── lecture1.ipynb              # Code from 2nd week of lectures
│   ├── ...
│   └── lecture7e.ipynb             # Code from 7th week of lectures
├── PracticeExams/                  # Exam prep materials
│   ├── 3dplotTest.ipynb/           # 3D rendering script for GPU testing
│   ├── practiceExam1-11.ipynb      # Midterm practice problems
│   ├── practiceExam1-12.ipynb      # Midterm practice problems
│   ├── practiceExam1-13.ipynb      # Midterm practice problems
│   ├── practiceExam1-14.ipynb      # Midterm practice problems
│   ├── practiceExam1-15.ipynb      # Midterm practice problems
│   └── quiz5.ipynb                 # Practice quiz question
├── assets/                         # Images, diagrams, and resources
└── README.md                       # This document
```

---

## 🧠 Topics Covered

### Fundamental Concepts

**1. Perceptron Algorithm**
- Single-layer perceptrons
- Linear separability
- Decision boundaries
- Weight updates and bias

**2. Neural Networks**
- Multi-layer perceptrons (MLPs)
- Activation functions (ReLU, sigmoid, tanh, softmax)
- Forward propagation
- Backpropagation algorithm

**3. Gradient Descent**
- Batch gradient descent
- Stochastic gradient descent (SGD)
- Mini-batch gradient descent
- Momentum and adaptive learning rates

### Advanced Architectures

**4. Convolutional Neural Networks (CNNs)**
- Convolution layers and kernels
- Pooling operations (max, average)
- Feature maps and filters
- Image classification tasks

**5. Deep Learning Techniques**
- Dropout regularization
- Batch normalization
- Transfer learning
- Data augmentation

**6. Model Optimization**
- Loss functions (MSE, cross-entropy)
- Optimizers (Adam, RMSprop, Adagrad)
- Learning rate scheduling
- Early stopping

---

## 🛠️ Technology Stack

| Technology | Purpose | Documentation |
|------------|---------|---------------|
| **Python 3.x** | Core programming language | [Python Docs](https://docs.python.org/3/) |
| **TensorFlow** | Deep learning framework | [TensorFlow](https://www.tensorflow.org/) |
| **Keras** | High-level neural network API | [Keras Docs](https://keras.io/) |
| **Jupyter Notebook** | Interactive development environment | [Jupyter](https://jupyter.org/) |
| **NumPy** | Numerical computations | [NumPy Docs](https://numpy.org/) |
| **Matplotlib** | Data visualization | [Matplotlib](https://matplotlib.org/) |
| **scikit-learn** | Machine learning utilities | [scikit-learn](https://scikit-learn.org/) |

---

## 🚀 Getting Started

### Prerequisites

**Python Environment:**
- Python 3.8 or higher
- Mambaforge package manager (recommended)
- Conda/Mamba environments

**Hardware Setup:**
- **Development Machine:** Dell Precision 5540 Laptop
  - Intel Core i9 processor
  - **NVIDIA Quadro T2000 (4GB VRAM)** - GPU acceleration for model training
  - CUDA-enabled TensorFlow for local GPU training
  - 16GB+ system RAM recommended
  - SSD storage for faster data loading

> 💡 **GPU Advantage:** All models in this repository were trained using the NVIDIA Quadro T2000, significantly reducing training time compared to CPU-only execution. TensorFlow automatically detects and utilizes the GPU when properly configured.

### Installation

**Using Mambaforge (Recommended):**

```bash
# Clone the repository
git clone https://github.com/ADolbyB/deep-learning-python.git
cd deep-learning-python

# Create conda environment with Python 3.10
mamba create -n deep-learning python=3.10
mamba activate deep-learning

# Install TensorFlow with GPU support
mamba install -c conda-forge tensorflow-gpu cudatoolkit cudnn

# Install additional packages
mamba install -c conda-forge keras numpy matplotlib jupyter scikit-learn pandas

# Verify GPU detection
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"

# Launch Jupyter Notebook or VS Code
jupyter notebook
# Or use VS Code with Jupyter extension
```

**Environment Location:**
- Conda environments stored at: `~/mambaforge/envs/deep-learning/`
- Package cache: `~/mambaforge/pkgs/`

**VS Code Setup (GPU-Accelerated Development):**

1. Install VS Code extensions:
   - Python
   - Jupyter
   - Pylance

2. Select the conda environment:
   - Press `Ctrl+Shift+P`
   - Type "Python: Select Interpreter"
   - Choose `~/mambaforge/envs/deep-learning/bin/python`

3. Open any `.ipynb` notebook and run cells with GPU acceleration

> 🎯 **Pro Tip:** Use `watch -n 1 nvidia-smi` in a separate terminal to monitor GPU utilization during training.

### Quick Start

1. **Navigate to Lectures** - Start with `Lectures/` for fundamentals
2. **Work through Assignments** - `Assignments/` are structured in order to follow assignments
3. **Review Practice Exams** - Test and modify to understand concepts
4. **Experiment** - Modify code and explore different approaches

---

## 📊 Sample Projects

### Assignment Highlights

**Perceptron Implementation**
- From-scratch perceptron algorithm
- Visualization of decision boundaries
- Binary classification problems

**Neural Network Training**
- Multi-layer network construction
- Custom training loops
- Performance evaluation and metrics

**CNN Image Classification**
- Image preprocessing pipelines
- Convolutional layer design
- Transfer learning with pre-trained models

---

## 🎓 Academic Context

**Course:** Introduction to Deep Learning  
**Level:** Upper-division Computer Science Elective  
**Format:** Jupyter Notebooks with embedded explanations and visualizations

**Learning Approach:**
- Theory combined with practical implementation
- Progressive difficulty from fundamentals to advanced topics
- Real-world datasets and problems
- Emphasis on understanding *why* algorithms work, not just *how*

---

## 📖 Key Learning Resources

### Official Documentation
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Keras Getting Started Guide](https://keras.io/getting_started/)
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) - Coursera

### Recommended Reading
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, Aaron Courville
- "Hands-On Machine Learning" by Aurélien Géron
- "Neural Networks and Deep Learning" by Michael Nielsen (free online)

### Video Resources
- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Sentdex: Deep Learning with Python](https://www.youtube.com/playlist?list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN)
- [Stanford CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)

---

## 💡 Best Practices Demonstrated

**Code Organization:**
- ✅ Modular, reusable functions
- ✅ Clear variable naming and documentation
- ✅ Proper train/validation/test splits
- ✅ Reproducible results with random seeds

**Model Development:**
- ✅ Baseline model establishment
- ✅ Iterative improvement and experimentation
- ✅ Hyperparameter tuning
- ✅ Performance visualization and analysis

**Documentation:**
- ✅ Markdown cells explaining concepts
- ✅ Inline comments for complex operations
- ✅ Visualizations of results and metrics
- ✅ Lessons learned and insights

---

## 🤝 Contributing

While this is primarily a coursework repository, improvements are welcome:

- 📝 Documentation enhancements
- 🐛 Bug fixes in implementations
- 💡 Additional examples or explanations
- 🎨 Visualization improvements

Please open an issue or submit a pull request!

---

## 📄 License

This project is licensed under the GNU GPL v3 License - see the [LICENSE.md](https://github.com/ADolbyB/deep-learning-python/blob/main/LICENSE.md) file for details.

**Academic Integrity Notice**: This repository represents completed coursework. If you're currently enrolled in a similar course, please use this as reference material only and adhere to your institution's academic honesty policies.

---

## 📧 Contact

**GitHub:** [Joel Brigida](https://github.com/ADolbyB)  
**LinkedIn:** [Joel Brigida](https://www.linkedin.com/in/joelmbrigida/)

Questions about implementations or concepts? Feel free to open an issue!

---

## 📊 Repository Stats

<div align="center">

![Repo Size](https://img.shields.io/github/repo-size/ADolbyB/deep-learning-python?style=for-the-badge&logo=github)
![Languages Count](https://img.shields.io/github/languages/count/ADolbyB/deep-learning-python?style=for-the-badge&logo=python)
![Top Language](https://img.shields.io/github/languages/top/ADolbyB/deep-learning-python?style=for-the-badge&logo=jupyter)
![Commits](https://img.shields.io/github/commit-activity/t/ADolbyB/deep-learning-python?style=for-the-badge&logo=github)

</div>

---

<div align="center">

**Master Deep Learning. Build Intelligent Systems. Transform Data into Insights.**

*From perceptrons to production-ready neural networks* 🧠

[![GitHub](https://img.shields.io/badge/Follow-ADolbyB-blue?style=for-the-badge&logo=github)](https://github.com/ADolbyB)

</div>