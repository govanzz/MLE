# Machine Learning Engineering (CS611)

This repository contains my coursework, assignments, and projects for **CS611: Machine Learning Engineering**, part of the **SMU MITB Programme**.

---

## Course Overview
This course equips me with the skills to handle the **engineering side of machine learning systems**.  
By the end of it, I will be able to manage the **end-to-end ML lifecycle** — from **data ingestion and model development** to **deployment, monitoring, and CI/CD (MLOps)**.  

I will gain the competency to build and run **production-grade ML systems on the cloud**, using open-source, cloud-compatible tools that mirror what is used in real-world industry settings.  

---

## My Learning Goals

Through this course, I aim to go beyond just applying ML models I will be **able to engineer complete ML systems** that can run reliably in real-world environments. Specifically, by the end of this course I will be able to:

- Strengthen my understanding of how to **take ML from experimentation to production**, not just stop at model accuracy.  
- Design **reproducible ML pipelines** that handle data ingestion, cleaning, feature engineering, and training in an automated way.  
- Work confidently with **Docker, orchestration tools, and CI/CD pipelines**, so that deploying ML models feels as natural as training them.  
- Apply **monitoring strategies** to detect model drift, data quality issues, and bias in deployed systems.  
- Extend **MLOps practices** not only to traditional ML models but also to **Generative AI** projects.  
- Think like an **engineer first, data scientist second** ensuring reliability, scalability, and maintainability in every project I build.

---
##  How to Run

This repository uses **Docker** and **Docker Compose** to ensure a consistent environment across all labs, assignments, and projects.

### 1️. Clone the Repository
```bash
git clone git@github.com:govanzz/MLE.git
cd MLE
```

### 2️. Navigate to a Lab/Assignment
Each lab, assignment, or project will have its own folder (e.g., `lab_1`, `assignment_1`, `project`).

```bash
cd lab_1
```
### 3️. Build the Docker Image

Run this once (or whenever the `Dockerfile` or `requirements.txt` changes):

```bash
docker-compose build
```

### 4️. Start the Environment

This launches JupyterLab and other services defined in `docker-compose.yml`:

```bash
docker-compose up -d
```
Then open JupyterLab at 👉 http://127.0.0.1:8888 or something similar

### 5️. Stop the Environment Safely

When finished, stop and remove containers:

```bash
docker-compose down
```

