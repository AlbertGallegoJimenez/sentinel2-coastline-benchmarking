# Comparative Evaluation of Satellite Image Segmentation Methods for Coastline Extraction: Spectral Indices vs. Deep Learning

This repository contains the codebase for my Master's Thesis (TFM) in Data Science at the **Universitat Oberta de Catalunya (UOC)**. The project focuses on benchmarking traditional remote sensing techniques against modern Deep Learning architectures for automated shoreline detection.

## 📌 Project Abstract
Accurate coastline extraction using high-frequency satellite data like Sentinel-2 is a cornerstone of modern coastal engineering, but current automated methods face limitations in environments with large intertidal zones, wave foam, and wet sand. 

This research assesses if state-of-the-art Deep Learning models can overcome the shortcomings of classical pixel-based remote sensing techniques by benchmarking them against established baselines (Otsu thresholding of NDWI/MNDWI).

## 🎯 Objectives
* **Primary Objective:** To perform a comprehensive benchmarking analysis between traditional spectral-based thresholding and Deep Learning-based semantic segmentation for coastline extraction.
* **Secondary Objectives:**
    * To generate a high-quality "Ground Truth" dataset of coastal masks using AI-assisted annotation (SAM).
    * To evaluate the performance degradation of each method under adverse conditions such as foam, differences in sand moisture, and clouds.
    * To assess the transferability of trained models between different coastal typologies (Cantabrian vs. Mediterranean).

## 🛠️ Methodology
The workflow is structured into four main phases:
1. **Data Acquisition:** Sentinel-2 Level-2A products from Google Earth Engine (2015-2025).
2. **Preprocessing & Labelling:** Calculation of spectral indices (NDWI, MNDWI) and Ground Truth generation using the Segment Anything Model (SAM).
3. **Model Implementation:** * **Baseline:** Automatic Otsu thresholding.
    * **Deep Learning:** Training models such as U-Net or SegFormer.
4. **Evaluation:** Dual validation using pixel-based metrics (IoU, Dice) and sub-pixel geometric validation (mean Euclidean distance).