#ifndef FINANCE_ML_H
#define FINANCE_ML_H

// Author: Dom G
// Finance ML Header - Class definitions for stock price prediction
// Enhanced with ROOT TTree integration

// Standard C++ includes
#include <vector>      // For std::vector
#include <string>      // For std::string
#include <memory>      // For std::unique_ptr
#include <iostream>    // For std::cout

// ROOT includes (for TTree functionality)
#include <TRandom3.h>  // ROOT random number generator
#include <TFile.h>     // ROOT file handling
#include <TTree.h>     // ROOT tree data structure
// #include <TMultiLayerPerceptron.h> // Neural network - not available in this ROOT version

// Model types enumeration (what kinds of models we can create)
enum ModelType {
    STOCK_PRICE_PREDICTION,
    VOLATILITY_PREDICTION,
    TREND_CLASSIFICATION
};

// Structure to hold financial features for ML
struct FinancialFeatures {
    std::vector<double> prices;      // Close prices
    std::vector<double> volumes;     // Trading volumes
    std::vector<double> rsi;         // RSI indicator
    std::vector<double> macd;        // MACD line
    std::vector<double> bollinger;   // Bollinger band position
    std::vector<double> volatility;  // Volatility measure
    std::vector<double> momentum;    // Momentum indicator
};

// Main ML Model Class
class FinanceMLModel {
private:
    // Private member variables (data that only this class can access)
    ModelType current_model_type; // from enum
    double learning_rate; // how fast the model learns
    int hidden_layers; // number of layers
    int training_epochs; // how many times to train on data
    std::unique_ptr<TRandom3> random_generator; // random number gen for seed

    // Feature data
    FinancialFeatures features; // calls features method
    std::vector<double> target_prices; // target prices vector
    std::vector<double> normalized_features; // normed features vector

    // Normalization parameters
    double feature_mean; // after norm, initialize mean
    double feature_std; // after norm, initialize std

    // Neural network (placeholder - TMultiLayerPerceptron not available in this ROOT version)
    // std::unique_ptr<TMultiLayerPerceptron> neural_network;

public:
    // Constructor and Destructor
    FinanceMLModel(ModelType type);
    ~FinanceMLModel();

    // Data loading method
    bool loadFromROOTFile(const std::string& filename);  //. Direct ROOT file loading

    // F/home/reddominick/Downloads/C++/Projects/Finance_Suite/Custom_LSTM.heature engineering methods
    void engineerFeatures();

    // Training and prediction methods
    void preprocessData();
    void trainModel(double train_split = 0.8);
    std::vector<double> predictStockPrices(const std::vector<double>& input_features, int days_ahead = 1);

    // Data access methods
    const FinancialFeatures& getFeatures() const { return features; }

    // Helper methods (will be implemented in Stock_ML.cpp)
    std::vector<double> calculateRSI(const std::vector<double>& prices, int period);
    std::vector<double> calculateMACD(const std::vector<double>& prices);
    std::vector<double> calculateBollingerBands(const std::vector<double>& prices, int period);

    // Neural network methods
    void initializeNeuralNetwork();
};

#endif // FINANCE_ML_H
