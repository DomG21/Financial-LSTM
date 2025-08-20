// Custom LSTM Implementation for Stock Prediction
// Author: Dom G
// Integrates with ROOT TTree data

// Header guards
#ifndef CUSTOM_LSTM_H
#define CUSTOM_LSTM_H

#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <TRandom3.h>

// Forward declarations
struct Matrix;
struct LSTMCell;

// Custom Matrix class for neural network operations
struct Matrix {
    std::vector<std::vector<double>> data;
    size_t rows, cols;

    Matrix(size_t r, size_t c) : rows(r), cols(c) {
        data.resize(rows, std::vector<double>(cols, 0.0));
    }

    // ========================================================================
    // MATRIX OPERATIONS
    // ========================================================================
    Matrix operator*(const Matrix& other) const;    // Matrix multiplication
    Matrix operator+(const Matrix& other) const;    // Matrix addition
    void randomize(double min = -1.0, double max = 1.0);  // Random initialization
    void zero();                                     // Zero all elements
    double& operator()(size_t i, size_t j) { return data[i][j]; }           // Access element
    const double& operator()(size_t i, size_t j) const { return data[i][j]; } // Const access
    void print() const;                              // Debug printing
};

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================
// Mathematical functions used in neural network computations
// ============================================================================
class Activation {
public:
    // Static lets us call these without creating an object
    // Sigmoid, Gate Controllers
    static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }  // Sigmoid: 0 to 1
    // Tanh, Value Processor
    static double tanh_func(double x) { return std::tanh(x); }               // Tanh: -1 to 1
    // ReLU (for future use)
    static double relu(double x) { return std::max(0.0, x); }               // ReLU: 0 or positive

    // Derivative functions for backpropagation
    static double sigmoid_derivative(double x) {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }

    static double tanh_derivative(double x) {
        double t = tanh_func(x);
        return 1.0 - t * t;
    }
};

// ============================================================================
// LSTM CELL IMPLEMENTATION
// ============================================================================
// Individual LSTM memory cell with gates and internal state
// Contains: Forget gate, Input gate, Output gate, Cell state, Hidden state
// ============================================================================
struct LSTMCell {
    // ========================================================================
    // LSTM GATE PARAMETERS
    // ========================================================================
    Matrix Wf, Wi, Wc, Wo;  // Input weight matrices (forget, input, candidate, output)
    Matrix Uf, Ui, Uc, Uo;  // Recurrent weight matrices for hidden state
    std::vector<double> bf, bi, bc, bo;  // Bias vectors for each gate

    // ========================================================================
    // LSTM INTERNAL STATES
    // ========================================================================
    std::vector<double> hidden_state;   // Short-term memory (output to next layer)
    std::vector<double> cell_state;     // Long-term memory (internal to cell)

    // ========================================================================
    // CELL CONFIGURATION
    // ========================================================================
    size_t input_size, hidden_size;     // Dimensions of input and hidden layers
    std::unique_ptr<TRandom3> rng;       // Random number generator for initialization

    // ========================================================================
    // CONSTRUCTOR
    // ========================================================================
    LSTMCell(size_t input_sz, size_t hidden_sz)
        : input_size(input_sz),
          hidden_size(hidden_sz),
          // Initialize weight matrices for all gates
          Wf(hidden_sz, input_sz), Wi(hidden_sz, input_sz),
          Wc(hidden_sz, input_sz), Wo(hidden_sz, input_sz),
          Uf(hidden_sz, hidden_sz), Ui(hidden_sz, hidden_sz),
          Uc(hidden_sz, hidden_sz), Uo(hidden_sz, hidden_sz),
          // Initialize bias vectors
          bf(hidden_sz, 0.0), bi(hidden_sz, 0.0),
          bc(hidden_sz, 0.0), bo(hidden_sz, 0.0),
          // Initialize states
          hidden_state(hidden_sz, 0.0), cell_state(hidden_sz, 0.0) {

        rng = std::make_unique<TRandom3>(42);  // Reproducible random seed
        initializeWeights();
    }

    // ========================================================================
    // WEIGHT INITIALIZATION
    // ========================================================================
    void initializeWeights() {
        // Xavier initialization for stable gradients
        double limit = std::sqrt(6.0 / (input_size + hidden_size));

        Wf.randomize(-limit, limit);   // Forget gate weights
        Wi.randomize(-limit, limit);   // Input gate weights
        Wc.randomize(-limit, limit);   // Candidate weights
        Wo.randomize(-limit, limit);   // Output gate weights

        Uf.randomize(-limit, limit);   // Recurrent weights
        Ui.randomize(-limit, limit);
        Uc.randomize(-limit, limit);
        Uo.randomize(-limit, limit);

        // Initialize forget gate bias to 1 (common LSTM trick)
        std::fill(bf.begin(), bf.end(), 1.0);
    }

    std::vector<double> forward(const std::vector<double>& input) {
        // LSTM forward pass
        std::vector<double> forget_gate(hidden_size);
        std::vector<double> input_gate(hidden_size);
        std::vector<double> candidate_gate(hidden_size);
        std::vector<double> output_gate(hidden_size);

        // Calculate gates
        for (size_t i = 0; i < hidden_size; i++) {
            double f_sum = bf[i];
            double i_sum = bi[i];
            double c_sum = bc[i];
            double o_sum = bo[i];

            // Input contributions
            for (size_t j = 0; j < input_size; j++) {
                f_sum += Wf(i, j) * input[j];
                i_sum += Wi(i, j) * input[j];
                c_sum += Wc(i, j) * input[j];
                o_sum += Wo(i, j) * input[j];
            }

            // Hidden state contributions
            for (size_t j = 0; j < hidden_size; j++) {
                f_sum += Uf(i, j) * hidden_state[j];
                i_sum += Ui(i, j) * hidden_state[j];
                c_sum += Uc(i, j) * hidden_state[j];
                o_sum += Uo(i, j) * hidden_state[j];
            }

            forget_gate[i] = Activation::sigmoid(f_sum);
            input_gate[i] = Activation::sigmoid(i_sum);
            candidate_gate[i] = Activation::tanh_func(c_sum);
            output_gate[i] = Activation::sigmoid(o_sum);
        }

        // Update cell state
        for (size_t i = 0; i < hidden_size; i++) {
            cell_state[i] = forget_gate[i] * cell_state[i] + input_gate[i] * candidate_gate[i];
        }

        // Update hidden state
        for (size_t i = 0; i < hidden_size; i++) {
            hidden_state[i] = output_gate[i] * Activation::tanh_func(cell_state[i]);
        }

        return hidden_state;
    }

    void reset() {
        std::fill(hidden_state.begin(), hidden_state.end(), 0.0);
        std::fill(cell_state.begin(), cell_state.end(), 0.0);
    }
};

// ============================================================================
// STOCK LSTM NEURAL NETWORK CLASS
// ============================================================================
// Multi-layer LSTM network specifically designed for stock price prediction
// Features: Deep architecture, backpropagation training, sequence processing
// ============================================================================
class StockLSTM {
private:
    // ========================================================================
    // NETWORK ARCHITECTURE COMPONENTS
    // ========================================================================
    std::vector<std::unique_ptr<LSTMCell>> layers;  // Stack of LSTM layers
    Matrix output_weights;                          // Final output layer weights
    std::vector<double> output_bias;                // Output layer bias

    // ========================================================================
    // NETWORK CONFIGURATION
    // ========================================================================
    size_t input_size;      // Number of input features (e.g., 5 for financial data)
    size_t hidden_size;     // Number of neurons per LSTM layer
    size_t num_layers;      // Number of LSTM layers (depth)
    size_t output_size;     // Number of outputs (1 for price prediction)

    double learning_rate;   // Training learning rate

public:
    // ========================================================================
    // CONSTRUCTOR
    // ========================================================================
    StockLSTM(size_t input_sz, size_t hidden_sz, size_t num_layers_val, size_t output_sz)
        : input_size(input_sz), hidden_size(hidden_sz),
          num_layers(num_layers_val), output_size(output_sz),
          output_weights(output_sz, hidden_sz), output_bias(output_sz, 0.0),
          learning_rate(0.001) {

        // Create LSTM layers
        for (size_t i = 0; i < num_layers; i++) {
            size_t layer_input_size = (i == 0) ? input_size : hidden_size;
            layers.push_back(std::make_unique<LSTMCell>(layer_input_size, hidden_size));
        }

        // Initialize output layer weights
        output_weights.randomize(-0.1, 0.1);
    }

    // ========================================================================
    // PREDICTION METHOD
    // ========================================================================
    // Process input sequence through all LSTM layers and generate prediction
    std::vector<double> predict(const std::vector<std::vector<double>>& sequence) {
        // Reset all layer states for new sequence
        for (auto& layer : layers) {
            layer->reset();
        }

        std::vector<double> layer_input;

        // Process sequence timestep by timestep
        for (const auto& timestep : sequence) {
            layer_input = timestep;

            // Forward through all LSTM layers
            for (auto& layer : layers) {
                layer_input = layer->forward(layer_input);
            }
        }

        // Final output layer (dense layer)
        std::vector<double> output(output_size, 0.0);
        for (size_t i = 0; i < output_size; i++) {
            output[i] = output_bias[i];
            for (size_t j = 0; j < hidden_size; j++) {
                output[i] += output_weights(i, j) * layer_input[j];
            }
        }

        return output;
    }

    // ========================================================================
    // TRAINING METHOD (Simplified)
    // ========================================================================
    // Train network using sequences and targets with simplified backpropagation
    void train(const std::vector<std::vector<std::vector<double>>>& sequences,
               const std::vector<std::vector<double>>& targets,
               int epochs = 100) {

        std::cout << "🧠 Training LSTM with " << sequences.size() << " sequences..." << std::endl;

        // Training loop over epochs
        for (int epoch = 0; epoch < epochs; epoch++) {
            double total_loss = 0.0;

            for (size_t seq_idx = 0; seq_idx < sequences.size(); seq_idx++) {
                std::vector<double> prediction = predict(sequences[seq_idx]);

                // Calculate MSE loss
                double loss = 0.0;
                for (size_t i = 0; i < output_size; i++) {
                    double error = prediction[i] - targets[seq_idx][i];
                    loss += error * error;
                }
                total_loss += loss;

                // Simple gradient update for output layer (simplified)
                for (size_t i = 0; i < output_size; i++) {
                    double error = prediction[i] - targets[seq_idx][i];
                    output_bias[i] -= learning_rate * error;

                    for (size_t j = 0; j < hidden_size; j++) {
                        output_weights(i, j) -= learning_rate * error * layers.back()->hidden_state[j];
                    }
                }
            }

            if (epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " << total_loss / sequences.size() << std::endl;
            }
        }

        std::cout << "✅ LSTM training complete!" << std::endl;
    }

    void setLearningRate(double lr) { learning_rate = lr; }
    double getLearningRate() const { return learning_rate; }
};

// ============================================================================
// MATRIX OPERATION IMPLEMENTATIONS
// ============================================================================

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::runtime_error("Matrix dimension mismatch for multiplication");
    }

    Matrix result(rows, other.cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < other.cols; j++) {
            for (size_t k = 0; k < cols; k++) {
                result(i, j) += data[i][k] * other.data[k][j];
            }
        }
    }
    return result;
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::runtime_error("Matrix dimension mismatch for addition");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result(i, j) = data[i][j] + other.data[i][j];
        }
    }
    return result;
}

void Matrix::randomize(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            data[i][j] = dis(gen);
        }
    }
}

void Matrix::zero() {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            data[i][j] = 0.0;
        }
    }
}

void Matrix::print() const {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            std::cout << data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

#endif // CUSTOM_LSTM_H
