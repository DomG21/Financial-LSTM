// Monte Carlo Stock Price Simulation using ROOT - DEBUG VERSION
// Author: Dom G
// This program predicts stock prices using Geometric Brownian Motion

#include "TCanvas.h"        // creates canvas for plotting
#include "TH1F.h"          // 1D histograms for data visualization
#include "TGraph.h"        // basic plots for price paths
#include "TMultiGraph.h"   // allows for multiple plots on same canvas
#include "TRandom3.h"      // random number generator
#include "TF1.h"           // 1D mathematical functions
#include "TLegend.h"       // plot legends
#include "TStyle.h"        // allows for colors, fonts, and other plot appearances
#include "TMath.h"         // mathematical operations
#include "TLatex.h"        // allows for text on plots and canvases
#include <iostream>        // input and output operations
#include <vector>          // vector data types
#include <fstream>         // file input/output operations
#include <sstream>         // string stream operations
#include <string>          // string literals and operations
#include <algorithm>       // computer science functions like find() and sort()
#include <iomanip>         // manipulation for formatting outputs
#include <cmath>           // for std::isnan and other math functions

using std::cout;           // shorthand for standard output
using std::endl;           // shorthand for end line
using std::vector;         // shorthand for vector data type
using std::string;         // shorthand for string data type

// Structure to hold stock data point - like a container for one row of stock data
struct StockData {
    string datetime;       // timestamp of the data point
    double open;          // first market price of the period
    double high;          // highest market price of the period
    double low;           // lowest market price of the period
    double close;         // last market price of the period
    long long volume;     // amount of shares traded that period
    double returns = 0.0; // calculated daily/period returns (percentage change)
};

// Monte Carlo prediction algorithm class - main engine for predictions
class AdaptiveMonteCarloPredictor {
private:
    vector<StockData> historical_data;  // stores all historical stock data
    string stock_symbol;                // ticker symbol (e.g., AAPL, MSFT)
    double historical_volatility;       // measure of price variability
    double historical_drift;            // average price trend (up/down)
    double current_price;              // most recent stock price

    // Monte Carlo simulation parameters
    int num_simulations;               // number of price paths to simulate
    int prediction_periods;            // how many time periods to predict forward
    bool is_intraday;                 // flag: true for minute data, false for daily data
    TRandom3* rng;                    // pointer to random number generator

public:
    // Constructor - like __init__() in Python, sets up the predictor
    AdaptiveMonteCarloPredictor(int simulations = 10000)
        : num_simulations(simulations), rng(new TRandom3(42)) {  // seed=42 for reproducible results
        gStyle->SetOptStat(1111);     // show name, entries, mean, RMS on histograms
        gStyle->SetPalette(1);        // use rainbow color palette
    }

    // Destructor - cleans up memory when object is destroyed
    ~AdaptiveMonteCarloPredictor() {
        delete rng;  // free the random number generator memory
    }

    // Main function to load CSV data and prepare for analysis
    bool loadCSVData(const string& filename) {
        cout << "Loading historical data from: " << filename << endl;

        // Try to open the CSV file
        std::ifstream file(filename);
        if (!file.is_open()) {
            cout << "Error: Could not open file " << filename << endl;
            return false;  // return false if file can't be opened
        }

        historical_data.clear();  // clear any existing data
        string line;              // temporary variable to hold each line
        bool first_line = true;   // flag to skip header row
        int line_count = 0;       // counter for debugging

        // Read file line by line
        while (getline(file, line)) {
            line_count++;

            // Skip the header row (first line)
            if (first_line) {
                cout << "Header: " << line << endl;  // show what header looks like
                first_line = false;
                continue;  // skip to next line
            }

            // Try to parse this line into a StockData structure
            StockData data;
            if (parseCSVLine(line, data)) {
                historical_data.push_back(data);  // add valid data to our collection
            } else {
                cout << "Warning: Failed to parse line " << line_count << ": " << line << endl;
            }
        }
        file.close();  // close the file when done

        // Check if we got any valid data
        if (historical_data.empty()) {
            cout << "Error: No valid data found!" << endl;
            return false;
        }

        cout << "Successfully loaded " << historical_data.size() << " data points" << endl;

        // Show first few data points for verification
        cout << "\n=== FIRST 3 DATA POINTS ===" << endl;
        for (size_t i = 0; i < std::min(size_t(3), historical_data.size()); i++) {
            cout << "Point " << i << ": " << historical_data[i].datetime
                 << " Close: $" << historical_data[i].close << endl;
        }

        // Show last few data points for verification
        cout << "\n=== LAST 3 DATA POINTS ===" << endl;
        size_t start = std::max(size_t(0), historical_data.size() - 3);
        for (size_t i = start; i < historical_data.size(); i++) {
            cout << "Point " << i << ": " << historical_data[i].datetime
                 << " Close: $" << historical_data[i].close << endl;
        }

        // Extract stock symbol from filename and analyze data
        extractStockSymbol(filename);     // get ticker from filename
        detectDataFrequency();           // determine if intraday or daily data
        calculateHistoricalStats();      // calculate volatility and drift

        return true;  // success!
    }

private:
    // Parse a single line of CSV data into StockData structure
    bool parseCSVLine(const string& line, StockData& data) {
        std::stringstream ss(line);  // create string stream for parsing
        string field;                // temporary variable for each field

        try {
            // Parse DateTime field
            if (!getline(ss, data.datetime, ',')) return false;

            // Parse Open price
            if (!getline(ss, field, ',')) return false;
            data.open = std::stod(field);  // convert string to double

            // Parse High price
            if (!getline(ss, field, ',')) return false;
            data.high = std::stod(field);

            // Parse Low price
            if (!getline(ss, field, ',')) return false;
            data.low = std::stod(field);

            // Parse Close price
            if (!getline(ss, field, ',')) return false;
            data.close = std::stod(field);

            // Parse Volume
            if (!getline(ss, field, ',')) return false;
            data.volume = std::stoll(field);  // convert string to long long

            // Validate that all prices are positive (basic sanity check)
            if (data.open <= 0 || data.high <= 0 || data.low <= 0 || data.close <= 0) {
                cout << "ERROR: Invalid prices in line: " << line << endl;
                return false;
            }

            // Check that high >= low (another sanity check)
            if (data.high < data.low) {
                cout << "ERROR: High < Low in line: " << line << endl;
                return false;
            }

            return true;  // parsing successful
        } catch (const std::exception& e) {
            cout << "ERROR: Exception parsing line: " << line << " - " << e.what() << endl;
            return false;  // parsing failed
        }
    }

    // Extract stock symbol from filename (e.g., "AAPL_data.csv" -> "AAPL")
    void extractStockSymbol(const string& filename) {
        size_t lastSlash = filename.find_last_of("/\\");  // find last directory separator
        size_t start = (lastSlash == string::npos) ? 0 : lastSlash + 1;  // start after last slash
        size_t end = filename.find('_', start);           // find underscore
        if (end == string::npos) end = filename.find('.', start);  // or find dot if no underscore

        stock_symbol = filename.substr(start, end - start);  // extract symbol
        std::transform(stock_symbol.begin(), stock_symbol.end(),
                      stock_symbol.begin(), ::toupper);     // convert to uppercase
        cout << "Extracted stock symbol: " << stock_symbol << endl;
    }

    // Detect whether data is intraday (minute-level) or daily
    void detectDataFrequency() {
        if (historical_data.size() < 2) {
            is_intraday = false;
            return;
        }

        // Simple heuristic: if more than 500 data points, probably intraday
        // More sophisticated version could parse datetime stamps
        is_intraday = historical_data.size() > 500;

        if (is_intraday) {
            prediction_periods = 78;  // Predict next trading day (6.5 hours * 12 five-minute intervals)
            cout << "Detected: Intraday data (predicting " << prediction_periods << " periods)" << endl;
        } else {
            prediction_periods = 252; // Predict next year (252 trading days)
            cout << "Detected: Daily data (predicting " << prediction_periods << " periods)" << endl;
        }
    }

    // Calculate historical volatility and drift from the data
    void calculateHistoricalStats() {
        if (historical_data.size() < 2) {
            cout << "ERROR: Need at least 2 data points for statistics!" << endl;
            return;
        }

        cout << "\n=== CALCULATING HISTORICAL STATISTICS ===" << endl;

        vector<double> returns;      // will store all calculated returns
        vector<double> valid_prices; // will store all valid prices

        // First pass: validate all prices and collect valid ones
        for (size_t i = 0; i < historical_data.size(); i++) {
            double price = historical_data[i].close;
            if (price <= 0 || std::isnan(price)) {  // check for invalid prices
                cout << "ERROR: Invalid price at index " << i << ": " << price << endl;
                continue;  // skip this price
            }
            valid_prices.push_back(price);  // add to valid prices list
        }

        if (valid_prices.size() < 2) {
            cout << "ERROR: Not enough valid prices!" << endl;
            return;
        }

        cout << "Valid prices: " << valid_prices.size() << " out of " << historical_data.size() << endl;

        // Calculate returns (percentage changes between consecutive prices)
        for (size_t i = 1; i < valid_prices.size(); i++) {
            double prev_price = valid_prices[i-1];  // previous price
            double curr_price = valid_prices[i];    // current price

            // Calculate return as (new_price - old_price) / old_price
            double return_val = (curr_price - prev_price) / prev_price;

            // Check for extreme returns (likely data errors or stock splits)
            if (abs(return_val) > 0.5) {  // 50% change in one period is suspicious
                cout << "WARNING: Extreme return at index " << i << ": "
                     << (return_val * 100) << "% (from $" << prev_price
                     << " to $" << curr_price << ")" << endl;
                // Skip extreme returns that are likely data errors
                continue;
            }

            returns.push_back(return_val);  // add valid return to our list

            // Show first few returns for debugging
            if (i <= 5) {
                cout << "Return " << i << ": " << (return_val * 100)
                     << "% (from $" << prev_price << " to $" << curr_price << ")" << endl;
            }
        }

        if (returns.empty()) {
            cout << "ERROR: No valid returns calculated!" << endl;
            return;
        }

        cout << "Total valid returns: " << returns.size() << endl;

        // Calculate mean return (drift) - average percentage change per period
        double sum_returns = 0;
        for (double ret : returns) {
            sum_returns += ret;  // sum all returns
        }
        historical_drift = sum_returns / returns.size();  // average return

        cout << "Raw drift (per period): " << historical_drift
             << " (" << (historical_drift * 100) << "%)" << endl;

        // Calculate volatility (standard deviation of returns)
        double sum_squared_diff = 0;
        for (double ret : returns) {
            double diff = ret - historical_drift;        // deviation from mean
            sum_squared_diff += diff * diff;             // square the deviation
        }

        if (returns.size() <= 1) {
            cout << "ERROR: Not enough returns for volatility calculation!" << endl;
            return;
        }

        // Calculate variance and then standard deviation (volatility)
        double variance = sum_squared_diff / (returns.size() - 1);  // sample variance
        cout << "Raw variance: " << variance << endl;

        if (variance < 0) {  // should never happen, but check anyway
            cout << "ERROR: Negative variance!" << endl;
            return;
        }

        historical_volatility = sqrt(variance);  // volatility = standard deviation
        cout << "Raw volatility (per period): " << historical_volatility
             << " (" << (historical_volatility * 100) << "%)" << endl;

        // Check for NaN values before proceeding
        if (std::isnan(historical_volatility) || std::isnan(historical_drift)) {
            cout << "ERROR: NaN in base calculations!" << endl;
            cout << "  Volatility: " << historical_volatility << endl;
            cout << "  Drift: " << historical_drift << endl;
            return;
        }

        if (historical_volatility <= 0) {
            cout << "ERROR: Zero or negative volatility!" << endl;
            return;
        }

        // Convert per-period statistics to annual statistics
        if (is_intraday) {
            // For intraday data: 78 periods per day * 252 trading days per year
            double periods_per_year = 78 * 252;
            cout << "Annualizing for intraday data (periods per year: " << periods_per_year << ")" << endl;

            historical_drift *= periods_per_year;           // annualize drift
            historical_volatility *= sqrt(periods_per_year); // annualize volatility (sqrt rule)

            // Check for unrealistic values after annualizing
            if (historical_volatility > 10.0) {  // 1000% annual volatility is unrealistic
                cout << "WARNING: Extremely high annualized volatility: "
                     << (historical_volatility * 100) << "%" << endl;
                cout << "Consider using daily data instead of intraday" << endl;
                // Cap volatility to prevent numerical issues
                historical_volatility = 2.0;  // cap at 200% annual volatility
                cout << "Volatility capped at 200%" << endl;
            }
        } else {
            // For daily data: 252 trading days per year
            cout << "Annualizing for daily data (252 trading days per year)" << endl;
            historical_drift *= 252;           // annualize drift
            historical_volatility *= sqrt(252); // annualize volatility
        }

        // Set current price to the most recent close price
        current_price = historical_data.back().close;

        // Final validation and summary
        cout << "\n=== FINAL STATISTICS ===" << endl;
        cout << "Current price: $" << std::fixed << std::setprecision(2) << current_price << endl;
        cout << "Annual drift: " << std::setprecision(4) << historical_drift
             << " (" << (historical_drift * 100) << "%)" << endl;
        cout << "Annual volatility: " << historical_volatility
             << " (" << (historical_volatility * 100) << "%)" << endl;

        // Final validation checks
        if (std::isnan(current_price)) cout << "ERROR: Current price is NAN!" << endl;
        if (std::isnan(historical_volatility)) cout << "ERROR: Volatility is NAN!" << endl;
        if (std::isnan(historical_drift)) cout << "ERROR: Drift is NAN!" << endl;
        if (historical_volatility <= 0) cout << "ERROR: Volatility is zero or negative!" << endl;
        if (current_price <= 0) cout << "ERROR: Current price is zero or negative!" << endl;
    }

public:
    // Main Monte Carlo simulation function
    void runPrediction() {
        if (historical_data.empty()) {
            cout << "Error: No historical data loaded!" << endl;
            return;
        }

        cout << "\n=== STARTING MONTE CARLO PREDICTION ===" << endl;

        // Validate all inputs before starting simulation
        if (std::isnan(current_price) || current_price <= 0) {
            cout << "ERROR: Invalid current price: " << current_price << endl;
            return;
        }

        if (std::isnan(historical_volatility) || historical_volatility <= 0) {
            cout << "ERROR: Invalid volatility: " << historical_volatility << endl;
            return;
        }

        if (std::isnan(historical_drift)) {
            cout << "ERROR: Invalid drift: " << historical_drift << endl;
            return;
        }

        // Calculate time step for simulation
        double dt = is_intraday ? (1.0 / (78 * 252)) : (1.0 / 252);  // fraction of year per period

        // Check if parameters might cause numerical problems
        double max_drift_term = abs(historical_drift * dt);                    // max drift per step
        double max_vol_term = historical_volatility * sqrt(dt) * 4;           // max volatility term (4 sigma)

        cout << "\n=== PARAMETER VALIDATION ===" << endl;
        cout << "Time step (dt): " << dt << endl;
        cout << "Max drift term per step: " << max_drift_term << endl;
        cout << "Max vol term per step (4σ): " << max_vol_term << endl;
        cout << "Max combined exponent: " << max_drift_term + max_vol_term << endl;

        // Warn if parameters might cause overflow in exp() function
        if (max_drift_term + max_vol_term > 1.0) {
            cout << "WARNING: Parameters may cause numerical instability!" << endl;
            cout << "Consider reducing volatility or using smaller time steps" << endl;
        }

        cout << "\n=== " << stock_symbol << " Monte Carlo Price Prediction ===" << endl;
        cout << "Current Price: $" << std::fixed << std::setprecision(2) << current_price << endl;
        cout << "Prediction Periods: " << prediction_periods << endl;
        cout << "Simulations: " << num_simulations << endl;

        // Test one simulation path first to catch problems early
        cout << "\n=== TESTING SINGLE PATH ===" << endl;
        double test_price = current_price;
        bool path_valid = true;

        for (int t = 1; t <= std::min(10, prediction_periods); t++) {
            double Z = rng->Gaus(0, 1);  // random normal variable

            // Geometric Brownian Motion formula components
            double drift_term = (historical_drift - 0.5 * historical_volatility * historical_volatility) * dt;
            double vol_term = historical_volatility * sqrt(dt) * Z;
            double exponent = drift_term + vol_term;

            cout << "Step " << t << ":" << endl;
            cout << "  Random Z = " << Z << endl;
            cout << "  Drift term = " << drift_term << endl;
            cout << "  Volatility term = " << vol_term << endl;
            cout << "  Total exponent = " << exponent << endl;

            // Check if exponent will cause overflow
            if (abs(exponent) > 10) {  // exp(10) ≈ 22000, exp(-10) ≈ 0.00005
                cout << "  ERROR: Exponent too large, will cause overflow!" << endl;
                path_valid = false;
                break;
            }

            // Apply Geometric Brownian Motion formula: S(t+1) = S(t) * exp(drift + volatility*random)
            double multiplier = exp(exponent);
            test_price = test_price * multiplier;

            cout << "  Price multiplier = " << multiplier << endl;
            cout << "  New price = $" << test_price << endl;

            // Check for invalid results
            if (std::isnan(test_price) || test_price <= 0) {
                cout << "  ERROR: Invalid price generated!" << endl;
                path_valid = false;
                break;
            }
        }

        if (!path_valid) {
            cout << "ERROR: Path generation failed. Check your parameters!" << endl;
            return;
        }

        cout << "Single path test PASSED. Proceeding with full simulation..." << endl;

        // Storage for simulation results
        vector<double> final_prices;    // final price from each simulation
        vector<TGraph*> sample_paths;   // first 50 price paths for plotting

        // Create histograms for results
        double price_range = current_price * 0.8;  // ±80% range for histogram
        TH1F* h_final = new TH1F("h_final",
                                Form("%s Final Price Distribution", stock_symbol.c_str()),
                                100,
                                current_price - price_range,
                                current_price + price_range);

        TH1F* h_returns = new TH1F("h_returns",
                                  "Predicted Returns Distribution",
                                  100, -0.2, 0.2);  // -20% to +20% returns

        cout << "\nRunning " << num_simulations << " Monte Carlo simulations..." << endl;

        // Main simulation loop
        for (int sim = 0; sim < num_simulations; sim++) {
            vector<double> prices;  // price path for this simulation
            vector<double> times;   // time points

            double S = current_price;  // start with current price
            prices.push_back(S);
            times.push_back(0);

            // Generate price path using Geometric Brownian Motion
            for (int t = 1; t <= prediction_periods; t++) {
                double Z = rng->Gaus(0, 1);  // standard normal random variable

                // GBM formula: dS = S * (μ * dt + σ * sqrt(dt) * Z)
                // Integrated form: S(t+1) = S(t) * exp((μ - σ²/2) * dt + σ * sqrt(dt) * Z)
                double drift_term = (historical_drift - 0.5 * historical_volatility * historical_volatility) * dt;
                double vol_term = historical_volatility * sqrt(dt) * Z;

                // Debug first simulation for first few steps
                if (sim == 0 && t <= 3) {
                    cout << "Simulation 0, Step " << t << ": Z=" << Z
                         << ", drift=" << drift_term
                         << ", vol=" << vol_term << endl;
                }

                S = S * exp(drift_term + vol_term);  // apply GBM formula

                // Check for problems in first simulation
                if (sim == 0 && t <= 3) {
                    cout << "  New price S: $" << S << endl;
                    if (std::isnan(S)) {
                        cout << "  ERROR: NaN detected at step " << t << endl;
                        return;
                    }
                }

                prices.push_back(S);
                times.push_back(t);
            }

            // Store final price and calculate return
            final_prices.push_back(S);
            h_final->Fill(S);  // add to histogram

            double total_return = (S - current_price) / current_price;  // percentage return
            h_returns->Fill(total_return);

            // Store first 50 paths for plotting
            if (sim < 50) {
                TGraph* g = new TGraph(times.size());
                for (size_t i = 0; i < times.size(); i++) {
                    g->SetPoint(i, times[i], prices[i]);  // add point to graph
                }
                sample_paths.push_back(g);
            }

            // Show progress every 10%
            if ((sim + 1) % (num_simulations / 10) == 0) {
                cout << "Progress: " << ((sim + 1) * 100 / num_simulations) << "%" << endl;
            }
        }

        cout << "Simulation complete! Analyzing results..." << endl;

        // Calculate and display statistics
        displayPredictionResults(final_prices, h_final);

        // Create visualizations - pass the final_prices vector for confidence intervals
        createPredictionPlots(sample_paths, h_final, h_returns, final_prices);

        cout << "\nPrediction complete! Check generated plots." << endl;
    }

private:
    // Calculate and display prediction statistics
    void displayPredictionResults(vector<double>& final_prices, TH1F* h_final) {
        // Sort prices for percentile calculations
        std::sort(final_prices.begin(), final_prices.end());

        // Calculate various statistics
        double mean_price = h_final->GetMean();           // average predicted price
        double std_price = h_final->GetStdDev();          // standard deviation
        double median_price = final_prices[num_simulations / 2];  // middle value

        // Confidence intervals
        double percentile_5 = final_prices[num_simulations * 0.05];   // 5th percentile (95% confidence lower bound)
        double percentile_10 = final_prices[num_simulations * 0.10];  // 10th percentile (90% confidence lower bound)
        double percentile_90 = final_prices[num_simulations * 0.90];  // 90th percentile (90% confidence upper bound)
        double percentile_95 = final_prices[num_simulations * 0.95];  // 95th percentile (95% confidence upper bound)

        // Count how many simulations resulted in gains
        int above_current = 0;
        for (double price : final_prices) {
            if (price > current_price) above_current++;
        }
        double prob_gain = (double)above_current / num_simulations * 100;  // probability of gain

        // Display comprehensive results
        cout << "\n=== PREDICTION RESULTS ===" << endl;
        cout << "Current Price: $" << std::fixed << std::setprecision(2) << current_price << endl;
        cout << "Predicted Mean Price: $" << mean_price << endl;
        cout << "Predicted Median Price: $" << median_price << endl;
        cout << "Standard Deviation: $" << std_price << endl;

        cout << "\n=== CONFIDENCE INTERVALS ===" << endl;
        cout << "90% Confidence Interval: $" << percentile_10 << " - $" << percentile_90 << endl;
        cout << "  - Lower bound (10th percentile): $" << percentile_10 << endl;
        cout << "  - Upper bound (90th percentile): $" << percentile_90 << endl;
        cout << "95% Confidence Interval: $" << percentile_5 << " - $" << percentile_95 << endl;
        cout << "  - Lower bound (5th percentile): $" << percentile_5 << endl;
        cout << "  - Upper bound (95th percentile): $" << percentile_95 << endl;

        cout << "\n=== PROBABILITY ANALYSIS ===" << endl;
        cout << "Probability of Gain: " << std::setprecision(1) << prob_gain << "%" << endl;
        cout << "Expected Return: " << std::setprecision(2)
             << ((mean_price - current_price) / current_price * 100) << "%" << endl;

        // Value at Risk calculations for different confidence levels
        double var_90 = current_price - percentile_10;  // 90% confidence VaR
        double var_95 = current_price - percentile_5;   // 95% confidence VaR
        cout << "\n=== VALUE AT RISK ANALYSIS ===" << endl;
        cout << "Value at Risk (90% confidence): $" << var_90
             << " (" << (var_90 / current_price * 100) << "%)" << endl;
        cout << "Value at Risk (95% confidence): $" << var_95
             << " (" << (var_95 / current_price * 100) << "%)" << endl;

        // Upside potential calculations
        double upside_90 = percentile_90 - current_price;  // 90% confidence upside
        double upside_95 = percentile_95 - current_price;  // 95% confidence upside
        cout << "\n=== UPSIDE POTENTIAL ANALYSIS ===" << endl;
        cout << "Upside Potential (90% confidence): $" << upside_90
             << " (" << (upside_90 / current_price * 100) << "%)" << endl;
        cout << "Upside Potential (95% confidence): $" << upside_95
             << " (" << (upside_95 / current_price * 100) << "%)" << endl;
    }

    // Create visualization plots
    void createPredictionPlots(vector<TGraph*>& sample_paths, TH1F* h_final, TH1F* h_returns, vector<double>& final_prices) {
        // Create main canvas with 4 subplots
        TCanvas* c1 = new TCanvas("c1",
                                 Form("%s Monte Carlo Price Prediction", stock_symbol.c_str()),
                                 1400, 900);
        c1->Divide(2, 2);  // 2x2 grid of plots

        // Plot 1: Sample prediction paths with confidence bands
        c1->cd(1);  // go to first subplot
        TMultiGraph* mg = new TMultiGraph();
        mg->SetTitle(Form("%s Price Prediction Paths;Time Periods;Price ($)", stock_symbol.c_str()));

        // First, create confidence bands from all simulation paths
        vector<vector<double>> all_paths(prediction_periods + 1);  // store all prices at each time step
        for (int t = 0; t <= prediction_periods; t++) {
            all_paths[t].resize(num_simulations);
        }

        // Collect all prices at each time step from stored sample paths and regenerate missing ones
        for (int sim = 0; sim < num_simulations; sim++) {
            vector<double> prices;
            double S = current_price;
            prices.push_back(S);

            // Generate price path for confidence calculation
            double dt = is_intraday ? (1.0 / (78 * 252)) : (1.0 / 252);
            for (int t = 1; t <= prediction_periods; t++) {
                double Z = rng->Gaus(0, 1);
                double drift_term = (historical_drift - 0.5 * historical_volatility * historical_volatility) * dt;
                double vol_term = historical_volatility * sqrt(dt) * Z;
                S = S * exp(drift_term + vol_term);
                prices.push_back(S);
            }

            // Store prices for this simulation
            for (int t = 0; t <= prediction_periods; t++) {
                all_paths[t][sim] = prices[t];
            }
        }

        // Calculate confidence bands at each time step
        vector<double> time_points, mean_path, upper_90, lower_90, upper_95, lower_95;

        for (int t = 0; t <= prediction_periods; t++) {
            // Sort prices at this time step
            std::sort(all_paths[t].begin(), all_paths[t].end());

            time_points.push_back(t);

            // Calculate statistics at this time step
            double mean_price = 0;
            for (double price : all_paths[t]) {
                mean_price += price;
            }
            mean_price /= num_simulations;
            mean_path.push_back(mean_price);

            // Calculate percentiles
            lower_95.push_back(all_paths[t][num_simulations * 0.025]);  // 2.5th percentile
            lower_90.push_back(all_paths[t][num_simulations * 0.05]);   // 5th percentile
            upper_90.push_back(all_paths[t][num_simulations * 0.95]);   // 95th percentile
            upper_95.push_back(all_paths[t][num_simulations * 0.975]);  // 97.5th percentile
        }

        // Create confidence band graphs
        TGraph* g_mean = new TGraph(time_points.size());
        TGraph* g_upper_90 = new TGraph(time_points.size());
        TGraph* g_lower_90 = new TGraph(time_points.size());
        TGraph* g_upper_95 = new TGraph(time_points.size());
        TGraph* g_lower_95 = new TGraph(time_points.size());

        for (size_t i = 0; i < time_points.size(); i++) {
            g_mean->SetPoint(i, time_points[i], mean_path[i]);
            g_upper_90->SetPoint(i, time_points[i], upper_90[i]);
            g_lower_90->SetPoint(i, time_points[i], lower_90[i]);
            g_upper_95->SetPoint(i, time_points[i], upper_95[i]);
            g_lower_95->SetPoint(i, time_points[i], lower_95[i]);
        }

        // Style the confidence bands
        g_upper_95->SetFillColor(kYellow);
        g_upper_95->SetFillStyle(1001);  // solid fill
        g_upper_95->SetLineColor(kYellow);
        g_upper_95->SetLineWidth(0);

        g_upper_90->SetFillColor(kGreen);
        g_upper_90->SetFillStyle(1001);  // solid fill
        g_upper_90->SetLineColor(kGreen);
        g_upper_90->SetLineWidth(0);

        // Add sample paths to multigraph (make them more transparent)
        for (size_t i = 0; i < std::min(size_t(20), sample_paths.size()); i++) {  // show fewer paths
            sample_paths[i]->SetLineColor(kBlue);
            sample_paths[i]->SetLineWidth(1);
            sample_paths[i]->SetLineStyle(1);
            sample_paths[i]->SetLineColorAlpha(kBlue, 0.3);  // make transparent
            mg->Add(sample_paths[i]);
        }

        // Add confidence bands to multigraph
        mg->Add(g_upper_95);
        mg->Add(g_lower_95);
        mg->Add(g_upper_90);
        mg->Add(g_lower_90);
        mg->Add(g_mean);

        mg->Draw("AFL3");  // A=axis, F=fill area, L=line, 3=filled area between graphs

        // Style the mean line
        g_mean->SetLineColor(kRed);
        g_mean->SetLineWidth(3);
        g_mean->Draw("L same");

        // Add current price line
        TF1* f_current = new TF1("f_current", Form("%f", current_price), 0, prediction_periods);
        f_current->SetLineColor(kBlack);     // black line for current price
        f_current->SetLineWidth(2);         // thick line
        f_current->SetLineStyle(2);         // dashed line
        f_current->Draw("same");            // draw on same plot

        // Add legend for paths plot
        TLegend* paths_legend = new TLegend(0.60, 0.15, 0.95, 0.35);
        paths_legend->AddEntry(g_mean, "Expected Path", "l");
        paths_legend->AddEntry(g_upper_90, "90% Confidence", "f");
        paths_legend->AddEntry(g_upper_95, "95% Confidence", "f");
        paths_legend->AddEntry(f_current, "Current Price", "l");
        paths_legend->SetBorderSize(1);
        paths_legend->SetFillColor(kWhite);
        paths_legend->SetTextSize(0.03);
        paths_legend->Draw();

        // Plot 2: Final price distribution histogram
        c1->cd(2);  // go to second subplot
        h_final->SetFillColor(kCyan);     // cyan fill
        h_final->SetFillStyle(3004);     // hatched pattern
        h_final->SetLineColor(kBlue);    // blue outline
        h_final->SetLineWidth(2);        // thick outline
        h_final->GetXaxis()->SetTitle("Predicted Final Price ($)");
        h_final->GetYaxis()->SetTitle("Frequency");
        h_final->Draw();

        // Calculate needed statistics for plotting
        double median_price = final_prices[num_simulations / 2];  // median value
        double percentile_10 = final_prices[num_simulations * 0.10];  // 10th percentile
        double percentile_90 = final_prices[num_simulations * 0.90];  // 90th percentile

        // Add vertical line showing mean prediction
        TF1* f_mean = new TF1("f_mean", Form("%f", h_final->GetMean()),
                             h_final->GetXaxis()->GetXmin(),
                             h_final->GetXaxis()->GetXmax());
        f_mean->SetLineColor(kRed);      // red line for mean
        f_mean->SetLineWidth(2);
        f_mean->Draw("same");

        // Create shaded confidence interval areas
        double x_min = h_final->GetXaxis()->GetXmin();
        double x_max = h_final->GetXaxis()->GetXmax();

        // 90% confidence interval shaded area (10th to 90th percentile)
        TBox* ci_90_box = new TBox(percentile_10, 0, percentile_90, h_final->GetMaximum() * 0.8);
        ci_90_box->SetFillColor(kGreen);     // green fill
        ci_90_box->SetFillStyle(3003);      // diagonal hatching pattern
        ci_90_box->SetLineColor(kGreen);    // green outline
        ci_90_box->SetLineWidth(2);
        ci_90_box->Draw("same");

        // 95% confidence interval shaded area (5th to 95th percentile) - outer bounds
        double percentile_5 = final_prices[num_simulations * 0.05];
        double percentile_95 = final_prices[num_simulations * 0.95];

        TBox* ci_95_box = new TBox(percentile_5, 0, percentile_95, h_final->GetMaximum() * 0.6);
        ci_95_box->SetFillColor(kYellow);    // yellow fill for 95% CI
        ci_95_box->SetFillStyle(3004);      // different hatching pattern
        ci_95_box->SetLineColor(kOrange);   // orange outline
        ci_95_box->SetLineWidth(2);
        ci_95_box->Draw("same");

        // Redraw histogram on top of boxes
        h_final->Draw("same");

        // Redraw mean line on top
        f_mean->Draw("same");

        // Add legend for the shaded areas
        TLegend* legend = new TLegend(0.60, 0.70, 0.95, 0.90);  // position legend
        legend->AddEntry(h_final, "Price Distribution", "f");
        legend->AddEntry(f_mean, "Expected Price", "l");
        legend->AddEntry(ci_90_box, "90% Confidence", "f");
        legend->AddEntry(ci_95_box, "95% Confidence", "f");
        legend->SetBorderSize(1);        // add border
        legend->SetFillColor(kWhite);    // white background
        legend->SetTextSize(0.03);       // smaller text
        legend->Draw();

        // Plot 3: Returns distribution histogram
        c1->cd(3);  // go to third subplot
        h_returns->SetFillColor(kGreen);   // green fill
        h_returns->SetFillStyle(3005);    // different hatch pattern
        h_returns->SetLineColor(kGreen);  // green outline
        h_returns->SetLineWidth(2);
        h_returns->GetXaxis()->SetTitle("Predicted Total Return");
        h_returns->GetYaxis()->SetTitle("Frequency");
        h_returns->Draw();

        // Plot 4: Statistics summary text
        c1->cd(4);  // go to fourth subplot
        TLatex* tex = new TLatex();  // text object for annotations
        tex->SetTextSize(0.05);      // set text size
        tex->SetTextAlign(12);       // left-aligned text

        // Add summary statistics as text on the plot
        tex->DrawLatex(0.1, 0.95, Form("%s Prediction Summary:", stock_symbol.c_str()));
        tex->DrawLatex(0.1, 0.85, Form("Current Price: $%.2f", current_price));
        tex->DrawLatex(0.1, 0.75, Form("Mean Prediction: $%.2f", h_final->GetMean()));
        tex->DrawLatex(0.1, 0.65, Form("Median Prediction: $%.2f", median_price));

        // Add confidence intervals to summary
        tex->SetTextColor(kGreen);       // green text for confidence intervals
        tex->DrawLatex(0.1, 0.55, "90% Confidence Interval:");
        tex->DrawLatex(0.1, 0.50, Form("  $%.2f - $%.2f", percentile_10, percentile_90));

        tex->SetTextColor(kBlack);       // back to black text
        tex->DrawLatex(0.1, 0.40, Form("Data Type: %s", is_intraday ? "Intraday" : "Daily"));
        tex->DrawLatex(0.1, 0.35, Form("Historical Vol: %.1f%%", historical_volatility * 100));
        tex->DrawLatex(0.1, 0.30, Form("Prediction Periods: %d", prediction_periods));
        tex->DrawLatex(0.1, 0.25, Form("Simulations: %d", num_simulations));

        tex->SetTextColor(kBlue);        // change text color to blue
        tex->DrawLatex(0.1, 0.15, "Geometric Brownian Motion Model");
        tex->SetTextColor(kGreen);       // green text for confidence note
        tex->DrawLatex(0.1, 0.10, "Green area: 90% confidence region");
        tex->SetTextColor(kOrange);      // orange text for 95% CI
        tex->DrawLatex(0.1, 0.05, "Yellow area: 95% confidence region");

        c1->Update();  // refresh the canvas

        // Save results to files
        string filename = stock_symbol + "_monte_carlo_prediction";
        c1->SaveAs((filename + ".png").c_str());  // save as PNG image
        c1->SaveAs((filename + ".pdf").c_str());  // save as PDF document

        cout << "Plots saved as " << filename << ".png and .pdf" << endl;
    }
};

// Main function - entry point of the program
int main(int argc, char* argv[]) {
    cout << "=== Adaptive Monte Carlo Stock Predictor (DEBUG VERSION) ===" << endl;
    cout << "Supports any CSV with format: DateTime,Open,High,Low,Close,Volume" << endl;
    cout << "This version includes extensive debugging to fix NaN issues" << endl;

    // Get CSV filename from command line argument or user input
    string csv_file;
    if(argc > 1) {
        // Use command line argument if provided
        csv_file = argv[1];
        cout << "\nUsing CSV file from command line: " << csv_file << endl;
    } else {
        // Fall back to user input if no argument provided
        cout << "\nEnter CSV filename (with path if needed): ";
        std::getline(std::cin, csv_file);  // read entire line including spaces
    }

    // Get number of simulations from command line or user input
    int num_simulations;
    if(argc > 2) {
        // Use second command line argument if provided
        num_simulations = std::stoi(argv[2]);
        cout << "Using number of simulations from command line: " << num_simulations << endl;
    } else {
        // Fall back to user input or default
        cout << "Enter number of simulations (default 10000, try 1000 for testing): ";
        string input;
        std::getline(std::cin, input);
        // Use default if user just pressed enter, otherwise convert to integer
        num_simulations = input.empty() ? 10000 : std::stoi(input);
    }

    // Create predictor object
    AdaptiveMonteCarloPredictor predictor(num_simulations);

    // Try to load data and run prediction
    if (predictor.loadCSVData(csv_file)) {
        cout << "\nData loaded successfully. Starting prediction..." << endl;
        predictor.runPrediction();  // run the Monte Carlo simulation
    } else {
        cout << "Failed to load data. Please check:" << endl;
        cout << "1. File path is correct" << endl;
        cout << "2. CSV format: DateTime,Open,High,Low,Close,Volume" << endl;
        cout << "3. All price values are positive numbers" << endl;
        cout << "4. No missing data in price columns" << endl;
        return 1;  // exit with error code
    }

    // Only wait for user input if running interactively (no command line args)
    if(argc <= 1) {
        cout << "\nPress Enter to exit...";
        std::cin.get();  // wait for user to press enter
    }
    return 0;        // successful exit
}
