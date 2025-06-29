# neural_network_simulator.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Neural Network Simulator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ActivationFunctions:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x):
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def linear(x):
        return x
    
    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)
    
    @staticmethod
    def step(x):
        return (x >= 0).astype(float)
    
    @staticmethod
    def step_derivative(x):
        return np.zeros_like(x)
    
    @staticmethod
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def elu_derivative(x, alpha=1.0):
        return np.where(x > 0, 1, alpha * np.exp(x))

class SimplePerceptron:
    def __init__(self, learning_rate=0.01, n_epochs=100, activation='step'):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.activation = activation
        self.weights = None
        self.bias = None
        self.training_history = {'loss': [], 'accuracy': []}
        
    def _get_activation_function(self):
        activation_map = {
            'sigmoid': ActivationFunctions.sigmoid,
            'tanh': ActivationFunctions.tanh,
            'relu': ActivationFunctions.relu,
            'leaky_relu': ActivationFunctions.leaky_relu,
            'linear': ActivationFunctions.linear,
            'step': ActivationFunctions.step,
            'elu': ActivationFunctions.elu
        }
        return activation_map[self.activation]
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.normal(0, 0.1, n_features)
        self.bias = 0
        
        activation_func = self._get_activation_function()
        
        for epoch in range(self.n_epochs):
            total_error = 0
            correct_predictions = 0
            
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction = activation_func(linear_output)
                
                if self.activation == 'step':
                    prediction = 1 if prediction >= 0.5 else 0
                elif self.activation in ['sigmoid', 'tanh']:
                    prediction = 1 if prediction >= 0.5 else 0
                else:
                    prediction = 1 if linear_output >= 0 else 0
                
                error = y[i] - prediction
                
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
                
                total_error += error**2
                if prediction == y[i]:
                    correct_predictions += 1
            
            mse = total_error / n_samples
            accuracy = correct_predictions / n_samples
            self.training_history['loss'].append(mse)
            self.training_history['accuracy'].append(accuracy)
    
    def predict(self, X):
        activation_func = self._get_activation_function()
        linear_output = np.dot(X, self.weights) + self.bias
        predictions = activation_func(linear_output)
        
        if self.activation == 'step':
            return (predictions >= 0.5).astype(int)
        elif self.activation in ['sigmoid', 'tanh']:
            return (predictions >= 0.5).astype(int)
        else:
            return (linear_output >= 0).astype(int)

class ADALINE:
    def __init__(self, learning_rate=0.01, n_epochs=100, activation='linear'):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.activation = activation
        self.weights = None
        self.bias = None
        self.training_history = {'loss': [], 'accuracy': []}
    
    def _get_activation_function(self):
        activation_map = {
            'sigmoid': ActivationFunctions.sigmoid,
            'tanh': ActivationFunctions.tanh,
            'relu': ActivationFunctions.relu,
            'leaky_relu': ActivationFunctions.leaky_relu,
            'linear': ActivationFunctions.linear,
            'step': ActivationFunctions.step,
            'elu': ActivationFunctions.elu
        }
        return activation_map[self.activation]
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.normal(0, 0.1, n_features)
        self.bias = 0
        
        activation_func = self._get_activation_function()
        
        for epoch in range(self.n_epochs):
            total_error = 0
            correct_predictions = 0
            
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction = activation_func(linear_output)
                
                error = y[i] - linear_output
                
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
                
                total_error += error**2
                
                if self.activation == 'linear':
                    binary_pred = 1 if linear_output >= 0 else 0
                else:
                    binary_pred = 1 if prediction >= 0.5 else 0
                
                if binary_pred == y[i]:
                    correct_predictions += 1
            
            mse = total_error / n_samples
            accuracy = correct_predictions / n_samples
            self.training_history['loss'].append(mse)
            self.training_history['accuracy'].append(accuracy)
    
    def predict(self, X):
        activation_func = self._get_activation_function()
        linear_output = np.dot(X, self.weights) + self.bias
        predictions = activation_func(linear_output)
        
        if self.activation == 'linear':
            return (linear_output >= 0).astype(int)
        elif self.activation in ['sigmoid', 'tanh']:
            return (predictions >= 0.5).astype(int)
        else:
            return (predictions >= 0).astype(int)

class MADALINE:
    def __init__(self, learning_rate=0.01, n_epochs=100, activation='sigmoid', n_hidden=5):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.activation = activation
        self.n_hidden = n_hidden
        self.weights_input_hidden = None
        self.weights_hidden_output = None
        self.bias_hidden = None
        self.bias_output = None
        self.training_history = {'loss': [], 'accuracy': []}
    
    def _get_activation_function(self):
        activation_map = {
            'sigmoid': (ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative),
            'tanh': (ActivationFunctions.tanh, ActivationFunctions.tanh_derivative),
            'relu': (ActivationFunctions.relu, ActivationFunctions.relu_derivative),
            'leaky_relu': (ActivationFunctions.leaky_relu, ActivationFunctions.leaky_relu_derivative),
            'linear': (ActivationFunctions.linear, ActivationFunctions.linear_derivative),
            'elu': (ActivationFunctions.elu, ActivationFunctions.elu_derivative)
        }
        return activation_map[self.activation]
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.weights_input_hidden = np.random.normal(0, 0.1, (n_features, self.n_hidden))
        self.weights_hidden_output = np.random.normal(0, 0.1, self.n_hidden)
        self.bias_hidden = np.zeros(self.n_hidden)
        self.bias_output = 0
        
        activation_func, activation_derivative = self._get_activation_function()
        
        for epoch in range(self.n_epochs):
            total_error = 0
            correct_predictions = 0
            
            for i in range(n_samples):
                hidden_input = np.dot(X[i], self.weights_input_hidden) + self.bias_hidden
                hidden_output = activation_func(hidden_input)
                
                output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
                final_output = activation_func(output_input)
                
                error = y[i] - final_output
                total_error += error**2
                
                output_delta = error * activation_derivative(output_input)
                
                self.weights_hidden_output += self.learning_rate * output_delta * hidden_output
                self.bias_output += self.learning_rate * output_delta
                
                hidden_error = output_delta * self.weights_hidden_output
                hidden_delta = hidden_error * activation_derivative(hidden_input)
                
                self.weights_input_hidden += self.learning_rate * np.outer(X[i], hidden_delta)
                self.bias_hidden += self.learning_rate * hidden_delta
                
                prediction = 1 if final_output >= 0.5 else 0
                if prediction == y[i]:
                    correct_predictions += 1
            
            mse = total_error / n_samples
            accuracy = correct_predictions / n_samples
            self.training_history['loss'].append(mse)
            self.training_history['accuracy'].append(accuracy)
    
    def predict(self, X):
        activation_func, _ = self._get_activation_function()
        predictions = []
        
        for i in range(X.shape[0]):
            hidden_input = np.dot(X[i], self.weights_input_hidden) + self.bias_hidden
            hidden_output = activation_func(hidden_input)
            
            output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
            final_output = activation_func(output_input)
            
            predictions.append(1 if final_output >= 0.5 else 0)
        
        return np.array(predictions)

def preprocess_data(df, target_column):
    """Preprocess the uploaded dataset"""
    # Create feature matrix and target vector
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Handle categorical features
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Ensure binary classification
    unique_targets = y.unique()
    if len(unique_targets) > 2:
        most_frequent = y.value_counts().index[0]
        y = (y == most_frequent).astype(int)
    else:
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Convert to numpy arrays if they aren't already
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values
    
    return X, y

def create_plots(model, y_test, y_pred, model_type):
    """Create visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_type} Training Results', fontsize=16, fontweight='bold')
    
    # Training Loss
    axes[0, 0].plot(model.training_history['loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Training Loss Over Epochs')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Mean Squared Error')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training Accuracy
    axes[0, 1].plot(model.training_history['accuracy'], 'g-', linewidth=2)
    axes[0, 1].set_title('Training Accuracy Over Epochs')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # Feature Importance
    if model_type in ["Simple Perceptron", "ADALINE"]:
        feature_importance = np.abs(model.weights)
        axes[1, 1].bar(range(len(feature_importance)), feature_importance, color='skyblue')
        axes[1, 1].set_title('Feature Importance (|Weights|)')
        axes[1, 1].set_xlabel('Feature Index')
        axes[1, 1].set_ylabel('Absolute Weight Value')
    else:
        all_weights = np.concatenate([model.weights_input_hidden.flatten(), 
                                    model.weights_hidden_output.flatten()])
        axes[1, 1].hist(all_weights, bins=20, alpha=0.7, color='lightcoral')
        axes[1, 1].set_title('Weight Distribution')
        axes[1, 1].set_xlabel('Weight Value')
        axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    return fig

# Main Streamlit App
def main():
    st.title("Neural Network Simulator")
    st.markdown("Perceptron, ADALINE, and MADALINE with Multiple Activation Functions")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", 
                               ["Dataset Upload", "Model Training", "Model Comparison", "Activation Analysis"])
    
    if page == "Dataset Upload":
        st.header("Dataset Upload and Preprocessing")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Dataset Preview:")
            st.dataframe(df.head())
            
            st.write("Dataset Information:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            target_column = st.selectbox("Select Target Column", df.columns.tolist())
            
            if st.button("Preprocess Dataset"):
                try:
                    X, y = preprocess_data(df, target_column)
                    
                    # Store in session state
                    st.session_state['X'] = X
                    st.session_state['y'] = y
                    st.session_state['feature_names'] = df.drop(target_column, axis=1).columns.tolist()
                    st.session_state['target_name'] = target_column
                    st.session_state['dataset_name'] = uploaded_file.name
                    
                    st.success("Dataset preprocessed successfully!")
                    st.write(f"Features shape: {X.shape}")
                    st.write(f"Target shape: {y.shape}")
                    st.write(f"Target classes: {np.unique(y)}")
                    
                    # Show sample of preprocessed data
                    st.write("Preprocessed Features (first 5 rows):")
                    feature_df = pd.DataFrame(X[:5], columns=st.session_state['feature_names'])
                    st.dataframe(feature_df)
                    
                except Exception as e:
                    st.error(f"Error preprocessing dataset: {str(e)}")
    
    elif page == "Model Training":
        st.header("Model Training")
        
        if 'X' not in st.session_state:
            st.warning("Please upload and preprocess a dataset first.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox("Select Model", 
                                    ["Simple Perceptron", "ADALINE", "MADALINE"])
            activation_function = st.selectbox("Select Activation Function",
                                             ["sigmoid", "tanh", "relu", "leaky_relu", "linear", "elu", "step"])
            
        with col2:
            learning_rate = st.slider("Learning Rate", 0.001, 1.0, 0.01, 0.001)
            n_epochs = st.slider("Number of Epochs", 10, 500, 100, 10)
            test_size = st.slider("Test Size (%)", 10, 50, 20, 5)
        
        if st.button("Train Model"):
            try:
                X = st.session_state['X']
                y = st.session_state['y']
                
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size/100, random_state=42, stratify=y
                )
                
                # Initialize model
                if model_type == "Simple Perceptron":
                    model = SimplePerceptron(learning_rate=learning_rate, 
                                           n_epochs=n_epochs, 
                                           activation=activation_function)
                elif model_type == "ADALINE":
                    model = ADALINE(learning_rate=learning_rate, 
                                  n_epochs=n_epochs, 
                                  activation=activation_function)
                elif model_type == "MADALINE":
                    model = MADALINE(learning_rate=learning_rate, 
                                   n_epochs=n_epochs, 
                                   activation=activation_function)
                
                # Train model
                with st.spinner("Training model..."):
                    model.fit(X_train, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                train_accuracy = accuracy_score(y_train, y_pred_train)
                test_accuracy = accuracy_score(y_test, y_pred_test)
                precision = precision_score(y_test, y_pred_test, average='binary', zero_division=0)
                recall = recall_score(y_test, y_pred_test, average='binary', zero_division=0)
                f1 = f1_score(y_test, y_pred_test, average='binary', zero_division=0)
                
                # Display metrics
                st.subheader("Performance Metrics")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Train Accuracy", f"{train_accuracy:.4f}")
                with col2:
                    st.metric("Test Accuracy", f"{test_accuracy:.4f}")
                with col3:
                    st.metric("Precision", f"{precision:.4f}")
                with col4:
                    st.metric("Recall", f"{recall:.4f}")
                with col5:
                    st.metric("F1-Score", f"{f1:.4f}")
                
                # Display plots
                fig = create_plots(model, y_test, y_pred_test, model_type)
                st.pyplot(fig)
                
                # Store results for comparison
                st.session_state['last_results'] = {
                    'model_type': model_type,
                    'activation': activation_function,
                    'metrics': {
                        'accuracy': test_accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
                }
                
                # Show detailed classification report
                st.subheader("Detailed Classification Report")
                report = classification_report(y_test, y_pred_test, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(4))
                
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
    
    elif page == "Model Comparison":
        st.header("Model Comparison")
        
        if 'X' not in st.session_state:
            st.warning("Please upload and preprocess a dataset first.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            activation_function = st.selectbox("Select Activation Function",
                                             ["sigmoid", "tanh", "relu", "leaky_relu", "linear", "elu"])
            learning_rate = st.slider("Learning Rate", 0.001, 1.0, 0.01, 0.001, key="comp_lr")
            
        with col2:
            n_epochs = st.slider("Number of Epochs", 10, 300, 100, 10, key="comp_epochs")
        
        if st.button("Compare All Models"):
            try:
                X = st.session_state['X']
                y = st.session_state['y']
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
                
                models = {
                    'Simple Perceptron': SimplePerceptron(learning_rate=learning_rate, 
                                                         n_epochs=n_epochs, 
                                                         activation=activation_function),
                    'ADALINE': ADALINE(learning_rate=learning_rate, 
                                      n_epochs=n_epochs, 
                                      activation=activation_function),
                    'MADALINE': MADALINE(learning_rate=learning_rate, 
                                        n_epochs=n_epochs, 
                                        activation=activation_function)
                }
                
                results = {}
                
                progress_bar = st.progress(0)
                for i, (name, model) in enumerate(models.items()):
                    st.write(f"Training {name}...")
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    results[name] = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
                        'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
                        'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
                        'training_history': model.training_history
                    }
                    progress_bar.progress((i + 1) / len(models))
                
                # Display comparison results
                st.subheader("Comparison Results")
                
                # Create comparison table
                comparison_df = pd.DataFrame(results).T
                comparison_df = comparison_df.drop('training_history', axis=1)
                st.dataframe(comparison_df.round(4))
                
                # Create comparison plots
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('Model Comparison Results', fontsize=16, fontweight='bold')
                
                # Accuracy comparison
                models_names = list(results.keys())
                accuracies = [results[name]['accuracy'] for name in models_names]
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                
                bars = axes[0, 0].bar(models_names, accuracies, color=colors)
                axes[0, 0].set_title('Model Accuracy Comparison')
                axes[0, 0].set_ylabel('Accuracy')
                axes[0, 0].set_ylim(0, 1)
                
                # Training loss comparison
                for i, name in enumerate(models_names):
                    axes[0, 1].plot(results[name]['training_history']['loss'], 
                                  label=name, linewidth=2, color=colors[i])
                axes[0, 1].set_title('Training Loss Comparison')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # Metrics comparison
                metrics = ['accuracy', 'precision', 'recall', 'f1']
                x = np.arange(len(models_names))
                width = 0.2
                
                for i, metric in enumerate(metrics):
                    values = [results[name][metric] for name in models_names]
                    axes[1, 0].bar(x + i*width, values, width, label=metric.capitalize())
                
                axes[1, 0].set_title('All Metrics Comparison')
                axes[1, 0].set_xlabel('Models')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].set_xticks(x + width * 1.5)
                axes[1, 0].set_xticklabels(models_names)
                axes[1, 0].legend()
                axes[1, 0].set_ylim(0, 1)
                
                # Training accuracy comparison
                for i, name in enumerate(models_names):
                    axes[1, 1].plot(results[name]['training_history']['accuracy'], 
                                  label=name, linewidth=2, color=colors[i])
                axes[1, 1].set_title('Training Accuracy Comparison')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Accuracy')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Best model
                best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
                st.success(f"Best performing model: {best_model} with accuracy: {results[best_model]['accuracy']:.4f}")
                
            except Exception as e:
                st.error(f"Error comparing models: {str(e)}")
    
    elif page == "Activation Analysis":
        st.header("Activation Function Analysis")
        
        if 'X' not in st.session_state:
            st.warning("Please upload and preprocess a dataset first.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox("Select Model", 
                                    ["Simple Perceptron", "ADALINE", "MADALINE"],
                                    key="analysis_model")
            learning_rate = st.slider("Learning Rate", 0.001, 1.0, 0.01, 0.001, key="analysis_lr")
            
        with col2:
            n_epochs = st.slider("Number of Epochs", 10, 200, 100, 10, key="analysis_epochs")
        
        if st.button("Analyze Activation Functions"):
            try:
                X = st.session_state['X']
                y = st.session_state['y']
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
                
                activation_functions = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'linear', 'elu']
                if model_type == "Simple Perceptron":
                    activation_functions.append('step')
                
                results = {}
                
                progress_bar = st.progress(0)
                for i, activation in enumerate(activation_functions):
                    st.write(f"Testing {activation}...")
                    
                    try:
                        if model_type == "Simple Perceptron":
                            model = SimplePerceptron(learning_rate=learning_rate, 
                                                   n_epochs=n_epochs, 
                                                   activation=activation)
                        elif model_type == "ADALINE":
                            model = ADALINE(learning_rate=learning_rate, 
                                          n_epochs=n_epochs, 
                                          activation=activation)
                        elif model_type == "MADALINE":
                            model = MADALINE(learning_rate=learning_rate, 
                                           n_epochs=n_epochs, 
                                           activation=activation)
                        
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        results[activation] = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
                            'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
                            'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
                            'training_history': model.training_history
                        }
                    except Exception as e:
                        st.warning(f"Error with {activation}: {str(e)}")
                        continue
                    
                    progress_bar.progress((i + 1) / len(activation_functions))
                
                if results:
                    # Display results
                    st.subheader("Activation Function Comparison")
                    
                    # Create results table
                    results_df = pd.DataFrame(results).T
                    results_df = results_df.drop('training_history', axis=1)
                    st.dataframe(results_df.round(4))
                    
                    # Create visualization
                    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                    fig.suptitle(f'Activation Function Analysis - {model_type}', fontsize=16, fontweight='bold')
                    
                    # Accuracy comparison
                    activations = list(results.keys())
                    accuracies = [results[act]['accuracy'] for act in activations]
                    
                    bars = axes[0, 0].bar(activations, accuracies, color='skyblue')
                    axes[0, 0].set_title('Activation Function Accuracy Comparison')
                    axes[0, 0].set_ylabel('Accuracy')
                    axes[0, 0].set_ylim(0, 1)
                    axes[0, 0].tick_params(axis='x', rotation=45)
                    
                    # Training loss comparison
                    for activation in activations:
                        axes[0, 1].plot(results[activation]['training_history']['loss'], 
                                      label=activation, linewidth=2)
                    axes[0, 1].set_title('Training Loss by Activation Function')
                    axes[0, 1].set_xlabel('Epoch')
                    axes[0, 1].set_ylabel('Loss')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # Metrics heatmap
                    metrics_data = []
                    for activation in activations:
                        metrics_data.append([
                            results[activation]['accuracy'],
                            results[activation]['precision'],
                            results[activation]['recall'],
                            results[activation]['f1']
                        ])
                    
                    metrics_df = pd.DataFrame(metrics_data, 
                                            index=activations, 
                                            columns=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
                    
                    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1, 0])
                    axes[1, 0].set_title('All Metrics Heatmap')
                    
                    # Training accuracy comparison
                    for activation in activations:
                        axes[1, 1].plot(results[activation]['training_history']['accuracy'], 
                                      label=activation, linewidth=2)
                    axes[1, 1].set_title('Training Accuracy by Activation Function')
                    axes[1, 1].set_xlabel('Epoch')
                    axes[1, 1].set_ylabel('Accuracy')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Best activation function
                    best_activation = max(results.keys(), key=lambda k: results[k]['accuracy'])
                    st.success(f"Best activation function: {best_activation} with accuracy: {results[best_activation]['accuracy']:.4f}")
                
                else:
                    st.error("No activation functions could be tested successfully.")
                    
            except Exception as e:
                st.error(f"Error in activation function analysis: {str(e)}")

if __name__ == "__main__":
    main()