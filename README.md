# Very-Early-Ventures-Beta

## Published Work
This project is part of research and work that has been published on [Mirror](https://mirror.xyz/veryearly.eth/nEpX1pMdASJ7VAbJI_pvhOQRkKD5YkBh81bzQZndpzI). The full article explains the context of this analysis and its relevance in the emerging `Web3` space.

## Introduction

This repository provides a tool for analyzing the `Alpha` and `Beta` of different crypto assets. The goal is to assess how each asset performs in relation to `Ethereum (ETH-USD)`, with a specific focus on rolling windows for different time periods.

## Alpha and Beta

`Alpha` and `Beta` are two important metrics in financial analysis:

- `Alpha` indicates the asset's performance relative to the market. Positive `Alpha` signifies that the asset is outperforming the market.

- `Beta` measures the asset's volatility compared to the market. A `Beta` greater than 1 indicates higher volatility than the market.

## Defining the Market

For this analysis, `Ethereum (ETH)` is used as the benchmark. The goal is to determine how different crypto assets behave relative to `Ethereum's` price movement over time, and to identify those with higher or lower `Beta` relative to `Ethereum`.

## Features

- **Beta Calculation**: Analyzes how each token correlates with `Ethereum (ETH-USD)`.

- **Rolling Windows**: Provides flexibility to calculate `Beta` and `Alpha` over different periods (60, 90, 180, and 365 days).

- **Visualization**: Interactive charts showing `Alpha` and `Beta` metrics for selected tokens over time.

- **Multiple Token Selection**: Allows users to select and compare multiple tokens against `Ethereum`.

## Installation

1. Clone the repository:

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the `Streamlit` app:
    ```bash
    streamlit run streamlit_app.py
    ```

## Usage

- **Select Period**: Choose the desired rolling window period for Beta and Alpha analysis (60, 90, 180, or 365 days).

- **Select Tokens**: Choose the tokens you wish to analyze from the list.

- **Visualize**: View interactive plots displaying the Beta and Alpha values for selected tokens relative to Ethereum.
