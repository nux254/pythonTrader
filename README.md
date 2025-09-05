# Algorithmic Trading Robots

This repository contains Python-based algorithmic trading robots designed to interact with MetaTrader 4/5 (MT4/MT5) terminals. The system facilitates real-time market data acquisition, order management, and execution of scalping strategies across multiple financial assets, incorporating machine learning for adaptive signal generation and parameter optimization.

## Features

*   **MetaTrader 4/5 Integration:** Seamless communication with MT4/MT5 terminals for live market data, order placement, modification, and cancellation.
*   **Real-time Data Processing:** Handles tick and bar data for multiple symbols, enabling robust real-time analysis.
*   **Scalping Strategies:** Implements short-term trading strategies based on a combination of technical indicators.
*   **Technical Indicator Suite:** Utilizes popular indicators like Average True Range (ATR), Exponential Moving Averages (EMA), Relative Strength Index (RSI), and Bollinger Bands for signal generation.
*   **Machine Learning Integration:** Employs a RandomForestClassifier for adaptive signal generation, enhancing strategy performance.
*   **Dynamic Position Sizing:** Calculates position sizes based on account balance and risk percentage, ensuring proper risk management.
*   **Trailing Stop Loss:** Automatically adjusts stop-loss orders to lock in profits as prices move favorably.
*   **Parameter Optimization:** Includes a mechanism for optimizing strategy parameters (e.g., Stop Loss coefficient, Take Profit/Stop Loss ratio) using historical data and walk-forward testing.
*   **Multi-Asset Support:** Designed to manage trading strategies across a portfolio of multiple currency pairs or commodities (e.g., XAUUSD, EURUSD, GBPUSD).
*   **Robust Logging:** Comprehensive logging for monitoring strategy execution, market events, and potential errors.
*   **Scheduled Operations:** Uses `APScheduler` to schedule regular strategy execution and periodic parameter optimization.

## Project Structure

*   `api/dwx_client.py`: The core client library for interfacing with MT4/MT5 terminals. It manages data exchange (market data, orders, historic data) via file-based communication.
*   `deep_refactor.py`: An implementation of a scalping strategy, focusing on technical indicators and trade execution.
*   `multipleAsset_chatgpt.py`: An advanced scalping strategy supporting multiple assets, incorporating a Machine Learning model for signal generation, dynamic position sizing, and parameter optimization.
*   `strategy.log`: Log file for strategy execution.
*   Other files: Various other Python scripts likely represent different iterations, scrapbooks, or alternative strategy implementations.

## Getting Started

### Prerequisites

*   Python 3.x
*   MetaTrader 4 or 5 terminal installed and configured.
*   The DWX-connector files (provided separately, usually MQL4/MQL5 files) need to be placed in your MetaTrader terminal's `MQL4/Files` or `MQL5/Files` directory. The `metatrader_dir_path` in the Python scripts should point to this directory.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Algorithmic-Trading-Robots.git
    cd Algorithmic-Trading-Robots
    ```
2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Note: A `requirements.txt` file is not present in the provided structure, but it's good practice to create one with `pandas`, `pandas-ta`, `scikit-learn`, `APScheduler`, and `backtesting`.)

### Configuration

*   **MetaTrader Path:** Update the `mt4_dir_path` variable in `deep_refactor.py` and `multipleAsset_chatgpt.py` (and any other relevant strategy files) to match the `MQL4/Files` or `MQL5/Files` directory of your MetaTrader terminal. For example:
    ```python
    mt4_dir_path='C:/Users/User/AppData/Roaming/MetaQuotes/Terminal/B7238334EF4B2B20A39D097B2877DE6E/MQL4/Files/'
    ```
*   **Symbols and Timeframe:** Adjust the `symbols` and `timeframe` variables within the strategy files (`deep_refactor.py`, `multipleAsset_chatgpt.py`) according to your trading preferences.
*   **Strategy Parameters:** Modify `risk_percentage`, `slatrcoef`, `tpsl_ratio`, `max_spread`, `min_volatility`, `min_adx`, and `use_limit_orders` as needed.

### Running the Strategies

To run a strategy (e.g., `multipleAsset_chatgpt.py`):

```bash
python multipleAsset_chatgpt.py
```

The script will connect to your MetaTrader terminal, subscribe to market data, and execute the trading strategy based on its defined schedule and logic.

## Contributing

Contributions are welcome! Please feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is open-source and available under the [MIT License](LICENSE).
