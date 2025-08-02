## 🩺 Pneumonia Detection App

Welcome to the **Pneumonia Detection App**, a full-stack web application designed to help in the preliminary diagnosis of pneumonia from chest X-ray images. This project demonstrates a complete end-to-end workflow, from a Python-based backend API for image analysis to a modern React frontend for a user-friendly interface.

| Badge | Status |
| :--- | :--- |
| ![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/therealshai/pneumonia-detection/backend-deploy.yml?label=Backend%20Build) | [![GitHub Actions Status](https://github.com/therealshai/pneumonia-detection/actions/workflows/backend-deploy.yml/badge.svg)](https://github.com/therealshai/pneumonia-detection/actions/workflows/backend-deploy.yml) |
| ![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/therealshai/pneumonia-detection/frontend-deploy.yml?label=Frontend%20Build) | [![GitHub Actions Status](https://github.com/therealshai/pneumonia-detection/actions/workflows/frontend-deploy.yml/badge.svg)](https://github.com/therealshai/pneumonia-detection/actions/workflows/frontend-deploy.yml) |

---

## 📖 Table of Contents
* [✨ Features](#-features)
* [🏛️ Architecture](#%ef%b8%8f-architecture)
* [🛠️ Getting Started](#%ef%b8%8f-getting-started)
* [☁️ Deployment](#%ef%b8%8f-deployment)
* [📂 Project Structure](#-project-structure)
* [🤝 Contributing](#-contributing)
* [📄 License](#-license)

---

## ✨ Features
* **Upload & Analyze:** A simple drag-and-drop interface for uploading chest X-ray images.
* **Backend API:** A Python backend that processes the uploaded image and returns a prediction.
* **Responsive UI:** A modern, responsive frontend built with React, Vite, and Shadcn UI.
* **Automated Deployment:** CI/CD pipelines with GitHub Actions for seamless deployment to Azure.

---

## 🏛️ Architecture
The application follows a simple decoupled architecture:
* **Frontend:** A React application served by Azure Static Web Apps. It handles all user interactions and sends requests to the backend.
* **Backend:** A Python API (e.g., Flask or FastAPI) hosted on Azure App Service. It receives the image data from the frontend, processes it, and returns the analysis result.

Communication between the frontend and backend is handled via REST API calls.

---

## 🛠️ Getting Started
Follow these steps to set up the project locally for development.

### Prerequisites
Make sure you have the following installed on your machine:
* [Node.js](https://nodejs.org/en/) (v18 or higher)
* [npm](https://www.npmjs.com/) or [Yarn](https://yarnpkg.com/)
* [Python](https://www.python.org/downloads/) (v3.8 or higher)

### 1. Clone the repository
```bash
git clone [https://github.com/therealshai/pneumonia-detection.git](https://github.com/therealshai/pneumonia-detection.git)
cd pneumonia-detection
```

### 2. Frontend Setup
Navigate to the `frontend` directory, install the dependencies, and start the development server.
```bash
cd frontend
npm install # or yarn install
npm run dev
```
The frontend will be available at `http://localhost:5173`.

### 3. Backend Setup
In a new terminal, navigate to the `backend` directory, create a Python virtual environment, install the required packages, and run the server.
```bash
cd backend
python -m venv venv
source venv/bin/activate # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
python main.py 
```
The backend API will be available at `http://localhost:5000`.

---

## ☁️ Deployment
This project uses **GitHub Actions** for an automated CI/CD pipeline to **Azure**.
* **Backend:** The `backend-deploy.yml` workflow deploys the Python backend to Azure App Service whenever changes are pushed to the `backend` directory on the `main` branch.
* **Frontend:** The `frontend-deploy.yml` workflow builds and deploys the React frontend to Azure Static Web Apps whenever changes are pushed to the `frontend` directory on the `main` branch.

---

## 📂 Project Structure
```
pneumonia-detection/
├── .github/
│   └── workflows/
│       ├── backend-deploy.yml
│       └── frontend-deploy.yml
├── backend/
│   ├── venv/
│   ├── app.py
│   ├── requirements.txt
│   └── ...
├── frontend/
│   ├── src/
│   │   ├── lib/
│   │   └── components/
│   ├── package.json
│   ├── vite.config.ts
│   └── ...
└── README.md
```

---

## 🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'feat: Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
