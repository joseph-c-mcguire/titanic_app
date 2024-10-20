# Titanic Survival Prediction App

A web application that predicts how likely you would have survived the Titanic disaster based on various features. This app uses a machine learning model to make predictions and visualizes the results using t-SNE plots.

## Features

- Predict survival likelihood based on user input features.
- Visualize the decision boundary and data points using t-SNE.
- Interactive and user-friendly interface built with Dash.

## Getting Started

### Prerequisites

- Docker
- Git

### Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/titanic_app.git
    cd titanic_app
    ```

2. **Build the Docker image:**

    ```sh
    docker build -t titanic_app .
    ```

3. **Run the Docker container:**

    ```sh
    docker run -p 8050:8050 titanic_app
    ```

4. **Access the application:**

    Open your web browser and go to `http://localhost:8050`.

## Usage

1. Enter the required features in the input fields.
2. Click the "Predict" button to see the prediction result.
3. The t-SNE plot will update to show the new data point and the decision boundary.

## Project Structure

- `app.py`: Main application file.
- `Dockerfile`: Docker configuration file.
- `requirements.txt`: Python dependencies.
- `data/`: Directory containing the model, t-SNE object, and configuration files.
- `models/`: Directory containing the trained model and t-SNE object.
- `README.md`: Project documentation.

## Deployment

To deploy this application to GitHub Pages using Docker and GitHub Actions, follow these steps:

1. **Create a GitHub repository** and push your code to the repository.
2. **Set up GitHub Actions** by creating a workflow file `.github/workflows/deploy.yml` with the following content:

    ```yaml
    name: Deploy to GitHub Pages

    on:
      push:
        branches:
          - main

    jobs:
      build:
        runs-on: ubuntu-latest

        steps:
        - name: Checkout code
          uses: actions/checkout@v2

        - name: Set up Docker Buildx
          uses: docker/setup-buildx-action@v1

        - name: Cache Docker layers
          uses: actions/cache@v2
          with:
            path: /tmp/.buildx-cache
            key: ${{ runner.os }}-buildx-${{ github.sha }}
            restore-keys: |
              ${{ runner.os }}-buildx-

        - name: Log in to Docker Hub
          uses: docker/login-action@v1
          with:
            username: ${{ secrets.DOCKER_USERNAME }}
            password: ${{ secrets.DOCKER_PASSWORD }}

        - name: Build and push Docker image
          run: |
            docker build -t ${{ secrets.DOCKER_USERNAME }}/titanic_app .
            docker push ${{ secrets.DOCKER_USERNAME }}/titanic_app

        - name: Deploy to GitHub Pages
          uses: peaceiris/actions-gh-pages@v3
          with:
            github_token: ${{ secrets.GITHUB_TOKEN }}
            publish_dir: ./public
    ```

3. **Add secrets to your GitHub repository**:
    - `DOCKER_USERNAME`: Your Docker Hub username.
    - `DOCKER_PASSWORD`: Your Docker Hub password.
    - `GITHUB_TOKEN`: GitHub token (automatically provided by GitHub Actions).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.