// Author: Dom G
// This program fetches data from yahoo finance, parses JSON, and creates ROOT TTree files
// Daily data with technical indicators for ML processing
// Root files are binary so they are faster then CSV/JSON and work with root operations

#include <iostream> // needed to basically use print
#include <string> // needed to use string variables
#include <vector> // needed for vectors
#include <fstream> // needed for files
#include <curl/curl.h> // for HTTP requests, like bash curl command
#include <json/json.h> // for parsing json, making it usable in c++
#include <ctime> // for time functions
#include <iomanip> // for formatting output (decimal places, etc.)
#include <cmath> // for mathematical calculations
#include <algorithm> // for min, max functions

// ROOT includes for TTree functionality
#include <TFile.h> // ROOT file handling
#include <TTree.h> // ROOT tree data structure
#include <TString.h> // ROOT string handling

// variables and methods for code
using std::cout;        // Output
using std::cin;         // Input
using std::endl;        // End line
using std::string;      // Text strings
using std::vector;      // Dynamic arrays
using std::ofstream;    // File output
using std::ifstream;    // File input
using std::min;         // Minimum function
using std::max;         // Maximum function

// Callback function to write HTTP response data
// Callback functions are like relays to other functions, this checks the size of the data
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
// *contents, point to contents
// size_t makes it so size is always positive integer value
// *userp = userpointer, points to what user wants
{
    ((string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb; // string points to user, and data is appended into char contents
}

// Structure to hold one day's financial data with technical indicators
struct FinancialDataPoint {
    string date;           // Date in YYYY-MM-DD format
    double open_price;     // Opening price
    double high_price;     // Highest price
    double low_price;      // Lowest price
    double close_price;    // Closing price
    long long volume;      // Trading volume
    time_t timestamp;      // Unix timestamp

    // Technical Indicators (calculated)
    double rsi_14;         // 14-day Relative Strength Index
    double macd_line;      // MACD line
    double macd_signal;    // MACD signal line
    double macd_histogram; // MACD histogram
    double bb_upper;       // Bollinger Band upper
    double bb_middle;      // Bollinger Band middle (SMA)
    double bb_lower;       // Bollinger Band lower
    double bb_position;    // Position within Bollinger Bands (0-1)
    double volatility_20;  // 20-day volatility
    double momentum_5;     // 5-day momentum
    double sma_20;         // 20-day Simple Moving Average
    double ema_12;         // 12-day Exponential Moving Average
    double ema_26;         // 26-day Exponential Moving Average
};

class YahooFinanceTTreeExtractor {
private:
    // private variables data isn't touched
    string ticker; // ticker is the stock symbol
    string responseData; // stores the JSON response from Yahoo Finance
    vector<FinancialDataPoint> financial_data; // stores parsed financial data

public:
    void setTicker(string t) {
        ticker = t; // void makes it return nothing, t is stored
    }

    bool fetchData() {
        cout << " Fetching chart data for: " << ticker << endl; // print what we're doing

        CURL *curl; // pointer to a curl object
        CURLcode res; // CURLcode is an enumerated data type in libcurl, used for error handling
        string readBuffer; // temporary storage for HTTP response data

        curl = curl_easy_init(); // another library function, creates the HTTP session
        if(curl) { // checks if CURL object was successfully created
            // Yahoo Finance API endpoint - gets more historical data
            string url = "https://query1.finance.yahoo.com/v8/finance/chart/" + ticker + "?range=1y&interval=1d";

            curl_easy_setopt(curl, CURLOPT_URL, url.c_str()); // session, OPTION, VALUE, value is a C-style string
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback); // set callback function for incoming data
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer); // set data storage location
            curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"); // need a "browser" to access HTTP protocol

            res = curl_easy_perform(curl); // finally executes the HTTP request
            curl_easy_cleanup(curl); // cleans up curl session afterwards

            if(res == CURLE_OK) { // part of CURLcode, checks if request was successful
                responseData = readBuffer; // copy temporary data to permanent storage
                cout << " Chart data fetched successfully!" << endl;
                return true; // success!
            } else {
                cout << " Failed to fetch data: " << curl_easy_strerror(res) << endl; // show error message
                return false; // failure
            }
        }
        return false; // if curl_easy_init() failed
    }

    void parseAndDisplayData() {
        if(responseData.empty()) { // checks if we have data to work with
            cout << " No data to parse!" << endl;
            return;
        }

        cout << "\n---  Stock Chart Data for " << ticker << " ---" << endl;

        // JSON parsing with json/json library
        Json::Value root; // container that holds the entire parsed JSON structure
        Json::Reader reader; // parser that converts JSON text into usable data structures

        if(reader.parse(responseData, root)) { // try to parse the JSON
            // Navigate the JSON structure safely
            if(root.isMember("chart") && root["chart"].isMember("result")) {
                Json::Value result = root["chart"]["result"][0];

                // Show basic current stock info
                if(result.isMember("meta")) {
                    Json::Value meta = result["meta"];
                    // Extract current price safely
                    if(meta.isMember("regularMarketPrice")) {
                        double price = meta["regularMarketPrice"].asDouble();
                        cout << " Current Price: $" << std::fixed << std::setprecision(2) << price << endl;
                    }
                    if(meta.isMember("currency")) {
                        string currency = meta["currency"].asString();
                        cout << " Currency: " << currency << endl;
                    }
                }

                // Count how many historical data points we got
                if(result.isMember("timestamp") && result.isMember("indicators")) {
                    Json::Value timestamps = result["timestamp"];
                    cout << " Historical data points available: " << timestamps.size() << endl;
                }
            }
        } else {
            cout << " Failed to parse JSON response!" << endl;
        }
    }

    /**
     * @brief Parse JSON data into structured format for ML
     * @return true if parsing successful, false otherwise
     */
    bool parseFinancialData() {
        cout << " Parsing financial data from JSON..." << endl;

        // Clear existing data
        financial_data.clear();

        // JSON parsing
        Json::Value root;
        Json::Reader reader;

        if(!reader.parse(responseData, root)) {
            cout << " Error: Could not parse JSON data!" << endl;
            return false;
        }

        // Check if the expected data structure exists
        if(!root.isMember("chart") || !root["chart"].isMember("result")) {
            cout << " Error: Invalid data structure!" << endl;
            return false;
        }

        Json::Value result = root["chart"]["result"][0];

        // Make sure we have historical data to work with
        if(!result.isMember("timestamp") || !result.isMember("indicators")) {
            cout << " Error: No historical data found!" << endl;
            return false;
        }

        // Extract the data arrays from JSON
        Json::Value timestamps = result["timestamp"]; // array of dates (as timestamps)
        Json::Value indicators = result["indicators"]["quote"][0]; // OHLCV data arrays

        // Process each day's data
        int dataPoints = timestamps.size();
        cout << " Processing " << dataPoints << " data points..." << endl;

        // Loop through each day's data
        for(int i = 0; i < dataPoints; i++) {
            FinancialDataPoint dataPoint;

            // Convert Unix timestamp to readable date
            time_t timestamp = timestamps[i].asInt64();
            dataPoint.timestamp = timestamp;

            struct tm* timeinfo = localtime(&timestamp);
            char dateStr[20];
            strftime(dateStr, sizeof(dateStr), "%Y-%m-%d", timeinfo);
            dataPoint.date = string(dateStr);

            // Extract OHLCV data (Open, High, Low, Close, Volume)
            // all from the Finance_ML.h file
            // Handle potential null values (some days might have missing data)
            dataPoint.open_price = indicators["open"][i].isNull() ? 0.0 : indicators["open"][i].asDouble();
            dataPoint.high_price = indicators["high"][i].isNull() ? 0.0 : indicators["high"][i].asDouble();
            dataPoint.low_price = indicators["low"][i].isNull() ? 0.0 : indicators["low"][i].asDouble();
            dataPoint.close_price = indicators["close"][i].isNull() ? 0.0 : indicators["close"][i].asDouble();
            dataPoint.volume = indicators["volume"][i].isNull() ? 0 : indicators["volume"][i].asInt64();

            // Initialize technical indicators (will be calculated later)
            dataPoint.rsi_14 = 0.0;
            dataPoint.macd_line = 0.0;
            dataPoint.macd_signal = 0.0;
            dataPoint.macd_histogram = 0.0;
            dataPoint.bb_upper = 0.0;
            dataPoint.bb_middle = 0.0;
            dataPoint.bb_lower = 0.0;
            dataPoint.bb_position = 0.0;
            dataPoint.volatility_20 = 0.0;
            dataPoint.momentum_5 = 0.0;
            dataPoint.sma_20 = 0.0;
            dataPoint.ema_12 = 0.0;
            dataPoint.ema_26 = 0.0;

            // Only add valid data points (skip null/zero data)
            if(dataPoint.close_price > 0) {
                financial_data.push_back(dataPoint);
            }
        }

        cout << " Successfully parsed " << financial_data.size() << " valid data points!" << endl;
        return true;
    }

    /**
     * @brief Calculate technical indicators for all data points
     */
    void calculateTechnicalIndicators() {
        cout << "🔧 Calculating technical indicators..." << endl;

        if(financial_data.size() < 30) {
            cout << "️ Warning: Need at least 30 data points for accurate technical indicators!" << endl;
            return;
        }

        // Calculate indicators for each data point
        for(size_t i = 0; i < financial_data.size(); i++) {
            // Calculate RSI (14-day)
            if(i >= 14) {
                financial_data[i].rsi_14 = calculateRSI(i, 14);
            }

            // Calculate Simple Moving Average (20-day)
            if(i >= 19) {
                financial_data[i].sma_20 = calculateSMA(i, 20);
            }

            // Calculate Exponential Moving Averages
            if(i >= 11) {
                financial_data[i].ema_12 = calculateEMA(i, 12);
            }
            if(i >= 25) {
                financial_data[i].ema_26 = calculateEMA(i, 26);
            }

            // Calculate MACD
            if(i >= 25 && financial_data[i].ema_12 != 0.0 && financial_data[i].ema_26 != 0.0) {
                financial_data[i].macd_line = financial_data[i].ema_12 - financial_data[i].ema_26;

                // Calculate MACD signal line (9-day EMA of MACD line)
                if(i >= 34) {
                    financial_data[i].macd_signal = calculateMACDSignal(i);
                    financial_data[i].macd_histogram = financial_data[i].macd_line - financial_data[i].macd_signal;
                }
            }

            // Calculate Bollinger Bands (20-day)
            if(i >= 19) {
                calculateBollingerBands(i);
            }

            // Calculate volatility (20-day)
            if(i >= 19) {
                financial_data[i].volatility_20 = calculateVolatility(i, 20);
            }

            // Calculate momentum (5-day)
            if(i >= 5) {
                financial_data[i].momentum_5 = calculateMomentum(i, 5);
            }
        }

        cout << " Technical indicators calculated!" << endl;
    }

    /**
     * @brief Calculate RSI for a given index
     */
    double calculateRSI(size_t index, int period) {
        if(index < period) return 0.0;

        double gains = 0.0, losses = 0.0;

        for(int i = 0; i < period; i++) {
            double change = financial_data[index - i].close_price - financial_data[index - i - 1].close_price;
            if(change > 0) gains += change;
            else losses += (-change);
        }

        double avg_gain = gains / period;
        double avg_loss = losses / period;

        if(avg_loss == 0.0) return 100.0;

        double rs = avg_gain / avg_loss;
        return 100.0 - (100.0 / (1.0 + rs));
    }

    /**
     * @brief Calculate Simple Moving Average
     */
    double calculateSMA(size_t index, int period) {
        if(index < period - 1) return 0.0;

        double sum = 0.0;
        for(int i = 0; i < period; i++) {
            sum += financial_data[index - i].close_price;
        }
        return sum / period;
    }

    /**
     * @brief Calculate Exponential Moving Average
     */
    double calculateEMA(size_t index, int period) {
        if(index < period - 1) return 0.0;

        double multiplier = 2.0 / (period + 1);

        if(index == period - 1) {
            // First EMA is SMA
            return calculateSMA(index, period);
        }

        double previous_ema = financial_data[index - 1].ema_12;
        if(period == 26) previous_ema = financial_data[index - 1].ema_26;

        return (financial_data[index].close_price * multiplier) + (previous_ema * (1 - multiplier));
    }

    /**
     * @brief Calculate MACD Signal Line
     */
    double calculateMACDSignal(size_t index) {
        if(index < 34) return 0.0;

        double multiplier = 2.0 / 10.0; // 9-day EMA

        if(index == 34) {
            // First signal is SMA of MACD values
            double sum = 0.0;
            for(int i = 0; i < 9; i++) {
                sum += financial_data[index - i].macd_line;
            }
            return sum / 9.0;
        }

        double previous_signal = financial_data[index - 1].macd_signal;
        return (financial_data[index].macd_line * multiplier) + (previous_signal * (1 - multiplier));
    }

    /**
     * @brief Calculate Bollinger Bands
     */
    void calculateBollingerBands(size_t index) {
        if(index < 19) return;

        double sma = financial_data[index].sma_20;
        financial_data[index].bb_middle = sma;

        // Calculate standard deviation
        double sum_squared_diff = 0.0;
        for(int i = 0; i < 20; i++) {
            double diff = financial_data[index - i].close_price - sma;
            sum_squared_diff += diff * diff;
        }
        double std_dev = sqrt(sum_squared_diff / 20.0);

        financial_data[index].bb_upper = sma + (2.0 * std_dev);
        financial_data[index].bb_lower = sma - (2.0 * std_dev);

        // Calculate position within bands (0-1 scale)
        double range = financial_data[index].bb_upper - financial_data[index].bb_lower;
        if(range > 0) {
            financial_data[index].bb_position =
                (financial_data[index].close_price - financial_data[index].bb_lower) / range;
        }
    }

    /**
     * @brief Calculate volatility
     */
    double calculateVolatility(size_t index, int period) {
        if(index < period - 1) return 0.0;

        vector<double> returns;
        for(int i = 0; i < period - 1; i++) {
            double return_val = log(financial_data[index - i].close_price / financial_data[index - i - 1].close_price);
            returns.push_back(return_val);
        }

        // Calculate standard deviation of returns
        double mean = 0.0;
        for(double ret : returns) mean += ret;
        mean /= returns.size();

        double sum_squared_diff = 0.0;
        for(double ret : returns) {
            sum_squared_diff += (ret - mean) * (ret - mean);
        }

        return sqrt(sum_squared_diff / (returns.size() - 1)) * sqrt(252); // Annualized
    }

    /**
     * @brief Calculate momentum
     */
    double calculateMomentum(size_t index, int period) {
        if(index < period) return 0.0;

        return (financial_data[index].close_price - financial_data[index - period].close_price)
               / financial_data[index - period].close_price * 100.0;
    }

    /**
     * @brief Save data to ROOT TTree file
     */
    void saveDataToTTree(const string& filename = "") {
        string output_filename = filename.empty() ? (ticker + "_financial_data.root") : filename;

        cout << " Saving financial data to ROOT TTree: " << output_filename << endl;

        // Create ROOT file
        TFile* file = new TFile(output_filename.c_str(), "RECREATE");
        if(!file || file->IsZombie()) {
            cout << "❌ Error: Could not create ROOT file!" << endl;
            return;
        }

        // Create TTree
        TTree* tree = new TTree("financial_data", ("Financial data for " + ticker).c_str());

        // TTree variables (must be persistent)
        char date_str[20]; // C-style character arrays for strings
        Double_t open_price, high_price, low_price, close_price;
        Long64_t volume, timestamp;
        Double_t rsi_14, macd_line, macd_signal, macd_histogram;
        Double_t bb_upper, bb_middle, bb_lower, bb_position;
        Double_t volatility_20, momentum_5, sma_20, ema_12, ema_26;

        // Create branches
        tree->Branch("date", date_str, "date/C");
        tree->Branch("timestamp", &timestamp, "timestamp/L");
        tree->Branch("open_price", &open_price, "open_price/D");
        tree->Branch("high_price", &high_price, "high_price/D");
        tree->Branch("low_price", &low_price, "low_price/D");
        tree->Branch("close_price", &close_price, "close_price/D");
        tree->Branch("volume", &volume, "volume/L");
        tree->Branch("rsi_14", &rsi_14, "rsi_14/D");
        tree->Branch("macd_line", &macd_line, "macd_line/D");
        tree->Branch("macd_signal", &macd_signal, "macd_signal/D");
        tree->Branch("macd_histogram", &macd_histogram, "macd_histogram/D");
        tree->Branch("bb_upper", &bb_upper, "bb_upper/D");
        tree->Branch("bb_middle", &bb_middle, "bb_middle/D");
        tree->Branch("bb_lower", &bb_lower, "bb_lower/D");
        tree->Branch("bb_position", &bb_position, "bb_position/D");
        tree->Branch("volatility_20", &volatility_20, "volatility_20/D");
        tree->Branch("momentum_5", &momentum_5, "momentum_5/D");
        tree->Branch("sma_20", &sma_20, "sma_20/D");
        tree->Branch("ema_12", &ema_12, "ema_12/D");
        tree->Branch("ema_26", &ema_26, "ema_26/D");

        // Fill TTree with data, root file expect C-style strings
        for(const auto& dataPoint : financial_data) {
            strcpy(date_str, dataPoint.date.c_str());
            timestamp = static_cast<Long64_t>(dataPoint.timestamp);
            open_price = dataPoint.open_price;
            high_price = dataPoint.high_price;
            low_price = dataPoint.low_price;
            close_price = dataPoint.close_price;
            volume = static_cast<Long64_t>(dataPoint.volume);
            rsi_14 = dataPoint.rsi_14;
            macd_line = dataPoint.macd_line;
            macd_signal = dataPoint.macd_signal;
            macd_histogram = dataPoint.macd_histogram;
            bb_upper = dataPoint.bb_upper;
            bb_middle = dataPoint.bb_middle;
            bb_lower = dataPoint.bb_lower;
            bb_position = dataPoint.bb_position;
            volatility_20 = dataPoint.volatility_20;
            momentum_5 = dataPoint.momentum_5;
            sma_20 = dataPoint.sma_20;
            ema_12 = dataPoint.ema_12;
            ema_26 = dataPoint.ema_26;

            tree->Fill();
        }

        // Write and close file
        tree->Write();
        file->Close();
        delete file;

        cout << " Financial TTree saved to: " << output_filename << endl;
        cout << " Ready for ML processing with Stock_ML.cpp!" << endl;
        cout << " Contains " << financial_data.size() << " entries with technical indicators!" << endl;
    }

    /**
     * @brief Display summary statistics of the data
     */
    void displayDataSummary() {
        if(financial_data.empty()) {
            cout << " No data available for summary!" << endl;
            return;
        }

        cout << "\n---  Data Summary for " << ticker << " ---" << endl;

        double min_price = financial_data[0].close_price;
        double max_price = financial_data[0].close_price;
        double total_volume = 0;

        for(const auto& dataPoint : financial_data) {
            min_price = std::min(min_price, dataPoint.close_price);
            max_price = std::max(max_price, dataPoint.close_price);
            total_volume += dataPoint.volume;
        }

        double avg_volume = total_volume / financial_data.size();
        double price_range = ((max_price - min_price) / min_price) * 100;

        cout << " Price Range: $" << std::fixed << std::setprecision(2) << min_price
             << " - $" << max_price << " (" << std::setprecision(1) << price_range << "% range)" << endl;
        cout << " Average Daily Volume: " << std::fixed << std::setprecision(0) << avg_volume << endl;
        cout << " Date Range: " << financial_data.front().date << " to " << financial_data.back().date << endl;
        cout << " Total Trading Days: " << financial_data.size() << endl;
    }

    /**
     * @brief Get parsed financial data for direct use
     * @return vector of FinancialDataPoint structures
     */
    const vector<FinancialDataPoint>& getFinancialData() const {
        return financial_data;
    }
};

int main(int argc, char* argv[]) {
    // Initialize curl globally - required before using any curl functions
    curl_global_init(CURL_GLOBAL_DEFAULT);

    YahooFinanceTTreeExtractor extractor; // creates new object called extractor

    cout << " Yahoo Finance ROOT TTree Data Extractor" << endl; // program header
    cout << "===========================================" << endl;
    cout << " Extracts daily financial data with technical indicators" << endl;
    cout << " Outputs ROOT TTree files ready for ML processing!" << endl << endl;

    // Get stock ticker from command line argument or user input
    string ticker;
    if(argc > 1) {
        // Use command line argument if provided
        ticker = argv[1];
        cout << " Using ticker from command line: " << ticker << endl;
    } else {
        // Fall back to user input if no argument provided
        cout << " Enter stock ticker (e.g., AAPL, TSLA): ";
        cin >> ticker; // cin converts input to the correct data type
    }

    extractor.setTicker(ticker); // send ticker to the extractor object

    if(extractor.fetchData()) { // try to get data from Yahoo Finance
        extractor.parseAndDisplayData(); // show basic info about the stock

        if(extractor.parseFinancialData()) { // parse JSON into structured format
            extractor.calculateTechnicalIndicators(); // calculate RSI, MACD, etc.
            extractor.displayDataSummary(); // show data statistics
            extractor.saveDataToTTree(); // save as ROOT TTree for ML processing

            cout << "\n Process complete!" << endl;
            cout << " ROOT TTree file is ready for machine learning!" << endl;
            cout << " Can be loaded directly by Stock_ML.cpp!" << endl;
        } else {
            cout << " Failed to parse financial data from JSON response." << endl;
        }
    } else {
        cout << " Failed to fetch data. Check internet connection and ticker symbol." << endl;
    }

    // Only wait for user input if running interactively (no command line args)
    if(argc <= 1) {
        cout << "\n️ Press Enter to exit...";
        cin.ignore(); // Clear any leftover input from previous cin
        cin.get();    // Wait for Enter key
    }

    // cleanup curl - required after using curl functions
    curl_global_cleanup();
    return 0; // program completed successfully
}
