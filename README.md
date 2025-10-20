# Project Insight Application

## ✨ Overview

This repository contains the source code for the **Project Insight Application**, a full-stack solution designed to **visualize real-time project performance metrics from our internal systems.**

The application is split into two components:

1. **Frontend (`psa-insight-ui`):** The user interface, built with modern JavaScript libraries.

2. **Backend (API):** A Python-based API providing data access and processing.

## 🚀 Getting Started

Follow these steps to get the application running on your local machine.

### Prerequisites

Ensure you have the following installed:

- Node.js (LTS version)

- Python 3.8+

- pip and virtual environment tools

### 💻 1. Frontend Setup

The frontend component is located in the `psa-insight-ui/` directory.

| Step  | Command                | Description                                                                      |
| ----- | ---------------------- | -------------------------------------------------------------------------------- |
| **1** | `cd psa-insight-ui`    | Navigate into the frontend directory.                                            |
| **2** | `npm install`          | Install all required Node.js dependencies.                                       |
| **3** | `mv .env.example .env` | Configure environment variables (API URL, etc.) by renaming the example file.    |
| **4** | `npm run dev`          | Start the development server. Access the application at `http://localhost:5173`. |

### ⚙️ 2. Backend Setup

The backend component requires a Python environment to run.

| Step  | Command                           | Description                                                                              |
| ----- | --------------------------------- | ---------------------------------------------------------------------------------------- |
| **1** | `pip install -r requirements.txt` | Install all necessary Python libraries.                                                  |
| **2** | `mv .env.example .env`            | Create the essential environment configuration file (Replace with appropriate API keys)  |
| **3** | `python3 app.py`                  | Run the Python application server. The API will be available at `http://127.0.0.1:8000`. |

## 🛠️ Built With

- **Frontend:** React

- **Backend:** Python, Flask, and Pandas
