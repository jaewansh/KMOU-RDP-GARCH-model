# KMOU-RDP-GARCH-model

Python code for reproducing the simulation and empirical results in the manuscript:

**Persistence Misspecification and Post-Crisis Volatility Overestimation: A Regime-Dependent GARCH Approach**

## Overview

This repository provides Python code for reproducing the main and supplementary results of the manuscript on post-crisis volatility dynamics.

The project investigates why standard single-regime GARCH models tend to overestimate volatility after large financial shocks and proposes a **Regime-Dependent Persistence GARCH (RDP-GARCH)** framework to address this problem. The core idea is that post-crisis volatility overestimation is driven by **persistence misspecification**, especially when persistence is averaged across heterogeneous volatility regimes.

The repository is intended to support transparent and reproducible academic research.

## Main Features

- Implementation of the **RDP-GARCH** model
- Controlled simulation experiments illustrating the mechanism of post-crisis volatility overestimation
- Empirical analyses using daily index data
- Supplementary figure generation for additional crisis episodes
- Reproducible Python workflow for estimation, visualization, and comparison

## Research Scope

This repository supports analyses related to:

- volatility overestimation after crisis periods,
- regime-dependent persistence in conditional variance dynamics,
- comparison between standard GARCH and RDP-GARCH,
- empirical applications to **KOSPI** and **S&P 500**,
- supplementary analyses for the **2008 Global Financial Crisis** and the **COVID-19 shock**.

## Repository Structure

```text
KMOU-RDP-GARCH-model/
├── code/
├── data/
│   └── figure/
│       └── supplement/
├── README.md
├── LICENSE
└── .gitignore
