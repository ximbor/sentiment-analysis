
# 📈 Sentiment Analysis Monitor  

The project's goal is to enhance and monitor social media reputation through automated sentiment analysis by implementing a continuous monitoring system and sentiment analysis model retraining.

**Solution Benefits**

1.  **Sentiment Analysis Automation**: By implementing a sentiment analysis model based on FastText, it will be possible to automate the processing of social media data to identify positive, neutral, and negative sentiments.
This will enable a rapid and targeted response to user feedback.
    
2.  **Continuous Reputation Monitoring**: Using MLOps methodologies, implement a continuous monitoring system to evaluate user sentiment trends over time.
This will allow for the quick detection of changes in company perception and prompt intervention if necessary.
    
3.  **Model Retraining**: Introducing an automated retraining system for the sentiment analysis model will ensure that the algorithm adapts dynamically to new data and shifts in language and user behavior on social media.
Maintaining high predictive accuracy is essential for a correct sentiment assessment.


## 🛠️ Endpoints

### 1. Readiness Check
Ensures that the service is up and, more importantly, that the machine learning model has been successfully loaded into memory.

-   **URL:** `/ready`    
-   **Method:** `GET`
-   **Success Response:**
    -   **Code:** 200 OK
    -   **Content:** `{"status": "ready"}`
        -   **Error Response:**
    -   **Code:** 503 Service Unavailable        
    -   **Content:** `{"detail": "Model loading..."}`        
    -   **Reason:** The model is still being initialized or failed to load.
        

### 2. Predict Sentiment
Analyzes the input text and returns the predicted sentiment label (Positive/Neutral/Negative) and a confidence score.

-   **URL:** `/predict`    
-   **Method:** `POST`    
-   **Request Body:**         
    ```
    {
      "text": "The new product launch was a great success!"
    }    
    ```
    
-   **Success Response:**
        -   **Code:** 200 OK
            -   **Content Example:**      
      ```
        {
          "text": "The new product launch was a great success!",
          "sentiment": "POSITIVE",
          "confidence": 0.9985
        }
       ```
        
-   **Error Response:**
    
    -   **Code:** 503 Service Unavailable
        
    -   **Content:** `{"detail": "Model not ready"}`  
  

## 🚀 CI/CD workflows

This section provides a detailed overview of the automated GitHub Actions workflows used to manage the testing, training, and deployment of the sentiment analysis system.

---

## 1. Pull Request Check (`pr-check.yaml`)
This workflow acts as a quality gate, ensuring that any code changes proposed via a Pull Request do not break existing functionality.

* **Triggers**:
    * **Pull Request**: Automatically runs when a PR is opened or updated targeting the `main` branch.
    * **Manual**: Can be triggered manually using the `workflow_dispatch` event for ad-hoc testing.
* **Key Job: Test**:
    * **Environment Validation**: Downloads the current production model (`ximbor/sentiment-monitor`) from Hugging Face to ensure the new code is compatible with existing model weights.
    * **Testing**: Executes the full test suite using `pytest` to validate API endpoints and inference logic.

---

## 2. Model Retraining and Deployment (`model-deploy.yaml`)
This workflow manages the Machine Learning operations (MLOps) cycle, specifically focusing on continuous training and model updates.

* **Triggers**:
    * **Scheduled**: Runs automatically every Monday at 03:00 UTC to incorporate new data.
    * **Manual**: Can be triggered manually via `workflow_dispatch`.
* **Key Jobs**:
    * **Train and Validate**: Installs ML dependencies, runs the `scripts/retrain.py` script, and validates the resulting model with `pytest`.
    * **Deploy**: If tests pass and the workflow is running on the `main` branch, it pushes the updated model to the **Hugging Face Model Hub**.

---

## 3. Application Deployment (`app-deploy.yaml`)
This workflow handles the Continuous Deployment (CD) of the FastAPI application to the production environment.

* **Triggers**:
    * **Push**: Triggered by any push to the `main` branch that modifies `main.py`, `Dockerfile`, or `requirements.txt`.
    * **Manual**: Can be triggered manually via `workflow_dispatch`.
* **Key Jobs**:
    * **Test**: Performs a final integration test of the application using the latest model from the Hub.
    * **Deploy**: Pushes the updated application code and configuration to **Hugging Face Spaces**.

---

### Workflow Architecture Diagram

The following Mermaid diagram visualizes how these workflows interact with each other and the Hugging Face ecosystem:

```mermaid
graph TD
    subgraph GitHub_Repo [GitHub Actions Workflows]
        PR[pr-check.yaml]
        M_WF[model-deploy.yaml]
        A_WF[app-deploy.yaml]
    end

    subgraph HF [Hugging Face Ecosystem]
        Hub[(Model Hub: sentiment-monitor)]
        Spaces[[Spaces: sentiment-analysis]]
    end

    %% PR Check Flow
    PR -->|Runs Validation| Tests_PR[pytest]
    Hub -.->|Downloads Model for Testing| Tests_PR

    %% Model Workflow Flow
    M_WF -->|Triggers Retraining| Train[scripts/retrain.py]
    Train -->|Pushes New Weights| Hub

    %% App Workflow Flow
    A_WF -->|Final Inference Test| Tests_App[pytest]
    Hub -.->|Downloads Model for Testing| Tests_App
    Tests_App -->|Deploys Application Code| Spaces

    %% Runtime Interaction
    Spaces -.->|Loads Model at Runtime| Hub
    ```