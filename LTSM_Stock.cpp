// Author: Dom G
// LSTM Stock Price Prediction Implementation
// Integrates Custom LSTM neural network with ROOT TTree data

#include "Finance_ML.h"
#include "Custom_LSTM.h"
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <iomanip>

/**
 * @brief LSTM-based Stock Predictor Class
 * Uses the custom LSTM implementation for time series prediction
 */
class LSTMStockPredictor {
private:
    std::unique_ptr<StockLSTM> lstm_model;
    FinanceMLModel ml_model;

    // Training parameters
    int sequence_length;
    int hidden_size;
    int num_layers;
    double learning_rate;

    // Processed data for LSTM
    std::vector<std::vector<std::vector<double>>> training_sequences;
    std::vector<std::vector<double>> training_targets;

public:
    /**
     * @brief Constructor - Initialize LSTM predictor
     */
    LSTMStockPredictor(int seq_len = 20, int hidden_sz = 50, int layers = 2)
        : ml_model(STOCK_PRICE_PREDICTION),
          sequence_length(seq_len),
          hidden_size(hidden_sz),
          num_layers(layers),
          learning_rate(0.001) {

        std::cout << " Initializing LSTM Stock Predictor..." << std::endl;
        std::cout << "   Sequence Length: " << sequence_length << std::endl;
        std::cout << "   Hidden Size: " << hidden_size << std::endl;
        std::cout << "   Layers: " << num_layers << std::endl;
    }

    /**
     * @brief Load financial data from ROOT file
     */
    bool loadData(const std::string& root_filename) {
        std::cout << " Loading data for LSTM training..." << std::endl;

        // Use the ML model to load ROOT data
        if (!ml_model.loadFromROOTFile(root_filename)) {
            std::cerr << "❌ Failed to load ROOT file!" << std::endl;
            return false;
        }

        // Engineer features (will preserve ROOT data)
        ml_model.engineerFeatures();

        std::cout << "✅ Data loaded successfully!" << std::endl;
        return true;
    }

    /**
     * @brief Prepare sequences for LSTM training
     */
    void prepareSequences() {
        std::cout << "🔄 Preparing LSTM training sequences..." << std::endl;

        // Get features from ML model
        const auto& features = getFeatures();

        // Check if we have enough data
        size_t min_size = std::min({
            features.prices.size(),
            features.rsi.size(),
            features.macd.size(),
            features.volatility.size(),
            features.momentum.size()
        });

        if (min_size < sequence_length + 1) {
            std::cerr << "❌ Not enough data for sequence creation!" << std::endl;
            return;
        }

        training_sequences.clear();
        training_targets.clear();

        // Create sequences
        for (size_t i = sequence_length; i < min_size - 1; i++) {
            std::vector<std::vector<double>> sequence;

            // Build sequence of feature vectors
            for (int t = 0; t < sequence_length; t++) {
                size_t idx = i - sequence_length + t;

                std::vector<double> feature_vector;

                // Normalize and add features
                if (idx < features.prices.size()) {
                    // Normalize price (simple min-max)
                    double norm_price = normalizePrice(features.prices[idx], features.prices);
                    feature_vector.push_back(norm_price);
                }

                if (idx < features.rsi.size()) {
                    feature_vector.push_back(features.rsi[idx] / 100.0); // RSI 0-100 -> 0-1
                }

                if (idx < features.macd.size()) {
                    feature_vector.push_back(tanh(features.macd[idx])); // Bound MACD
                }

                if (idx < features.volatility.size()) {
                    feature_vector.push_back(std::min(features.volatility[idx] * 10, 1.0)); // Scale volatility
                }

                if (idx < features.momentum.size()) {
                    feature_vector.push_back(tanh(features.momentum[idx] * 10)); // Bound momentum
                }

                sequence.push_back(feature_vector);
            }

            // Target is next day's price (normalized)
            std::vector<double> target;
            if (i + 1 < features.prices.size()) {
                double norm_target = normalizePrice(features.prices[i + 1], features.prices);
                target.push_back(norm_target);
            }

            training_sequences.push_back(sequence);
            training_targets.push_back(target);
        }

        // Initialize LSTM with correct input size
        int input_size = training_sequences[0][0].size();
        lstm_model = std::make_unique<StockLSTM>(input_size, hidden_size, num_layers, 1);
        lstm_model->setLearningRate(learning_rate);

        std::cout << "✅ Created " << training_sequences.size() << " training sequences" << std::endl;
        std::cout << "   Input features per timestep: " << input_size << std::endl;
    }

    /**
     * @brief Train the LSTM model
     */
    void trainModel(int epochs = 100) {
        if (training_sequences.empty() || !lstm_model) {
            std::cerr << "❌ No training data or LSTM model!" << std::endl;
            return;
        }

        std::cout << "🏋️ Training LSTM model..." << std::endl;

        // Train the LSTM
        lstm_model->train(training_sequences, training_targets, epochs);

        std::cout << "✅ LSTM training completed!" << std::endl;
    }

    /**
     * @brief Make price predictions
     */
    std::vector<double> predictPrices(int days_ahead = 5) {
        if (!lstm_model || training_sequences.empty()) {
            std::cerr << "❌ No trained model available!" << std::endl;
            return {};
        }

        std::cout << "🔮 Making LSTM predictions for " << days_ahead << " days..." << std::endl;

        // Use last sequence as starting point
        std::vector<std::vector<double>> current_sequence = training_sequences.back();
        std::vector<double> predictions;

        const auto& features = getFeatures();

        for (int day = 0; day < days_ahead; day++) {
            // Predict next value
            std::vector<double> prediction = lstm_model->predict(current_sequence);

            // Denormalize prediction
            double predicted_price = denormalizePrice(prediction[0], features.prices);
            predictions.push_back(predicted_price);

            // Update sequence for next prediction
            // Remove first timestep and add predicted timestep
            current_sequence.erase(current_sequence.begin());

            // Create new feature vector with prediction
            std::vector<double> new_features;
            new_features.push_back(prediction[0]); // Normalized predicted price

            // For other features, use simple extrapolation or last values
            if (current_sequence.size() > 0) {
                auto last_features = current_sequence.back();
                for (size_t i = 1; i < last_features.size(); i++) {
                    new_features.push_back(last_features[i]); // Use last known technical indicators
                }
            }

            current_sequence.push_back(new_features);

            std::cout << "   Day " << (day + 1) << ": $" << std::fixed << std::setprecision(2)
                      << predicted_price << std::endl;
        }

        return predictions;
    }

    /**
     * @brief Get features from ML model (helper function)
     */
    const FinancialFeatures& getFeatures() const {
        // This requires making features public in FinanceMLModel or adding a getter
        // For now, we'll need to modify the class structure
        return ml_model.getFeatures();
    }

private:
    /**
     * @brief Normalize price using min-max normalization
     */
    double normalizePrice(double price, const std::vector<double>& prices) {
        auto minmax = std::minmax_element(prices.begin(), prices.end());
        double min_price = *minmax.first;
        double max_price = *minmax.second;

        if (max_price == min_price) return 0.5; // Avoid division by zero

        return (price - min_price) / (max_price - min_price);
    }

    /**
     * @brief Denormalize price from 0-1 range back to actual price
     */
    double denormalizePrice(double norm_price, const std::vector<double>& prices) {
        auto minmax = std::minmax_element(prices.begin(), prices.end());
        double min_price = *minmax.first;
        double max_price = *minmax.second;

        return norm_price * (max_price - min_price) + min_price;
    }
};

/**
 * @brief Example usage function
 */
void demonstrateLSTMPrediction(const std::string& root_file) {
    std::cout << "=== LSTM Stock Prediction Demo ===" << std::endl;

    // Create LSTM predictor
    LSTMStockPredictor predictor(20, 64, 3); // 20-day sequences, 64 hidden units, 3 layers

    // Load data
    if (!predictor.loadData(root_file)) {
        std::cerr << "Failed to load data!" << std::endl;
        return;
    }

    // Prepare training sequences
    predictor.prepareSequences();

    // Train model
    predictor.trainModel(50); // 50 epochs

    // Make predictions
    std::vector<double> predictions = predictor.predictPrices(7); // Predict 1 week ahead

    std::cout << "\n=== Prediction Summary ===" << std::endl;
    for (size_t i = 0; i < predictions.size(); i++) {
        std::cout << "Day " << (i + 1) << ": $" << std::fixed << std::setprecision(2)
                  << predictions[i] << std::endl;
    }
}
