# Image Processing & API Integration Service

## Description
This project is a Flask-based Python service designed for image processing and API integration. It provides functionality for fetching images from an external API, processing them (e.g., resizing, cropping), and serving the processed images via a RESTful API. 

The application incorporates caching mechanisms, configurable parameters through environment variables, and health-check capabilities to ensure robust and efficient performance.

---

## Features
- **Image Fetching & Processing**:
  - Retrieves images from an external server (via the Immich API) based on user queries.
  - Processes images with cropping and resizing ensuring compatibility with 4K resolution.
- **Caching**:
  - Implements intelligent caching for API responses, reducing redundant calls to the server.
  - Supports expiration and cleanup of outdated cache entries.
- **REST Endpoints**:
  - `/new-image`: Retrieves, processes, and saves a new random image based on a query.
  - `/image`: Provides the most recently processed image.
  - `/cache-status`: Displays cache statistics and expiry info.
  - `/images-info`: Reports details about the processed images directory.
  - `/health`: Offers a health status of the service, checking system resources and API accessibility.
- **Error Handling**:
  - Provides user-friendly error reporting for API, image processing, and system-related issues.

---

## Requirements

### Dependencies
This project uses the following libraries:
- `Flask`: Web framework for creating RESTful endpoints.
- `requests`: For making HTTP API calls.
- `Pillow` and `pillow_heif`: For working with images.
- `OpenCV (cv2)`: For advanced image processing.
- `Pydantic`: For data validation and configuration.
- `dotenv`: For loading environment variables.
- `psutil`: For system resource monitoring.
- `numpy`: For array manipulation in image processing.

### Environment Variables
The following environment variables can be used to configure the service:
- `IMMICH_URL`: Base URL of the Immich API (must end with `/api`).
- `IMMICH_API_KEY`: API key for authenticating requests.
- `DEFAULT_QUERY`: Default search query for fetching images.

---

## Setup Instructions
### Local Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with the required environment variables:
   ```
   IMMICH_URL=https://example.com/api
   IMMICH_API_KEY=your_api_key
   DEFAULT_QUERY=Mountains
   ```
4. Run the application:
   ```bash
   python main.py
   ```

### Docker Setup
1. Build the Docker image:
   ```bash
   docker build -t image-processing-service .
   ```
2. Run the Docker container:
   ```bash
   docker run -p 5000:5000 --env-file .env image-processing-service
   ```

---

## API Endpoints

### GET `/new-image`
- **Description**: Fetches a new image, processes it, and saves it.
- **Parameters**:
  - `query` (optional): String representing the search query.
  - `range` (optional): Integer specifying the result limit.

---

### GET `/image`
- **Description**: Returns the latest processed image.

---

### GET `/cache-status`
- **Description**: Provides statistics about the current cache state.

---

### GET `/images-info`
- **Description**: Returns information about the images directory, including the saved images.

---

### GET `/health`
- **Description**: Health check endpoint to verify system readiness.

---

## Notes
- The `images` directory is automatically created in the project root for saving processed images.
- Cache expires after **3 hours** (configurable in the source code).

---

## License
This project is open-source and available under the [MIT License](LICENSE).