# Standard library imports
import os
import json
import time
import random
import logging
import hashlib
from functools import wraps
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from enum import Enum
import shutil

# Third-party library imports
import requests
from requests import ConnectionError
from flask import Flask, send_file, jsonify, request
from PIL import Image
import numpy as np
import cv2
from pydantic import BaseModel, Field, HttpUrl, field_validator
from dotenv import load_dotenv
import psutil


load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory setup
IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)


# Custom Exceptions
class ImageProcessingError(Exception):
	"""Base exception for image processing errors"""
	pass


class APIError(Exception):
	"""Exception for API-related errors"""
	pass


class ImageNotFoundError(Exception):
	"""Exception for missing image files"""
	pass


# Cache Configuration
CACHE_EXPIRY_SECONDS = 3 * 60 * 60  # 3 hours
_api_cache: Dict[str, Dict[str, Any]] = {}


def get_cache_key(url: str, method: str = "GET", data: Optional[Dict] = None, headers: Optional[Dict] = None, params: Optional[Dict] = None) -> str:
	"""
	Generate a unique cache key for API requests.

	Args:
		url (str): The API endpoint URL
		method (str, optional): HTTP method. Defaults to "GET"
		data (Optional[Dict], optional): Request data for POST requests. Defaults to None
		headers (Optional[Dict], optional): Request headers. Defaults to None
		params (Optional[Dict], optional): URL parameters. Defaults to None

	Returns:
		str: MD5 hash of the request parameters for use as cache key
	"""
	key_data = f"{method}:{url}"
	if data:
		key_data += f":data:{json.dumps(data, sort_keys=True)}"
	if headers:
		# Sort headers to ensure consistent key generation
		sorted_headers = json.dumps(dict(sorted(headers.items())), sort_keys=True)
		key_data += f":headers:{sorted_headers}"
	if params:
		key_data += f":params:{json.dumps(params, sort_keys=True)}"
	return hashlib.md5(key_data.encode()).hexdigest()



def is_cache_valid(cache_entry: Dict[str, Any]) -> bool:
	"""
	Check if a cache entry is still valid based on expiry time.

	Args:
		cache_entry (Dict[str, Any]): Cache entry containing timestamp and data

	Returns:
		bool: True if cache entry is still valid, False if expired
	"""
	return time.time() - cache_entry["timestamp"] < CACHE_EXPIRY_SECONDS


def cached_request(method: str, url: str, **kwargs) -> requests.Response:
	"""
	Wrapper for requests that caches responses for 3 hours.
	Excludes 'original' endpoints where actual image data is downloaded.

	This function implements intelligent caching for API requests:
	- Caches JSON responses for non-image endpoints
	- Bypasses cache for '/original' endpoints to ensure fresh image data
	- Automatically handles cache expiry and cleanup

	Args:
		method (str): HTTP method (GET, POST, etc.)
		url (str): Target URL for the request
		**kwargs: Additional arguments passed to requests.request()

	Returns:
		requests.Response: Either cached response or fresh API response

	Raises:
		requests.RequestException: For network or HTTP errors
	"""
	# Don't cache original image downloads
	if "/original" in url:
		logger.info(f"Bypassing cache for original image request: {url}")
		return requests.request(method, url, **kwargs)

	# Generate cache key including headers and other parameters
	data = kwargs.get('json') if method.upper() == 'POST' else kwargs.get('data')
	headers = kwargs.get('headers')
	params = kwargs.get('params')
	cache_key = get_cache_key(url, method, data, headers, params)


	# Check if we have a valid cached response
	if cache_key in _api_cache and is_cache_valid(_api_cache[cache_key]):
		logger.info(f"Using cached response for: {url}")
		cached_entry = _api_cache[cache_key]

		# Create a mock response object with cached data
		response = requests.Response()
		response._content = json.dumps(cached_entry["data"]).encode()
		response.status_code = cached_entry["status_code"]
		response.headers.update(cached_entry["headers"])
		return response

	# Make the actual request
	logger.info(f"Making fresh API request to: {url}")
	response = requests.request(method, url, **kwargs)

	# Cache successful responses (excluding binary content)
	if response.status_code == 200:
		try:
			# Only cache JSON responses
			response_data = response.json()
			_api_cache[cache_key] = {
				"data": response_data,
				"status_code": response.status_code,
				"headers": dict(response.headers),
				"timestamp": time.time()
			}
			logger.info(f"Cached response for: {url}")
		except (json.JSONDecodeError, ValueError):
			# Don't cache non-JSON responses
			logger.info(f"Not caching non-JSON response for: {url}")

	return response


def clear_expired_cache():
	"""
	Remove expired entries from the cache to prevent memory bloat.

	This function iterates through all cache entries and removes those
	that have exceeded the CACHE_EXPIRY_SECONDS threshold.
	"""
	current_time = time.time()
	expired_keys = [
		key for key, entry in _api_cache.items()
		if current_time - entry["timestamp"] >= CACHE_EXPIRY_SECONDS
	]

	for key in expired_keys:
		del _api_cache[key]

	if expired_keys:
		logger.info(f"Cleared {len(expired_keys)} expired cache entries")


# Pydantic Models
class ImageFormat(str, Enum):
	"""Supported image output formats."""
	PNG = "png"
	JPEG = "jpeg"
	HEIC = "heic"


class SearchQuery(BaseModel):
	"""Model for Immich search API requests."""
	query: str = Field(..., min_length=1, description="Search query string")
	type: str = Field("IMAGE", description="Asset type to search for")

	@field_validator('query')
	def validate_query(cls, v):
		# Add query sanitization
		forbidden_chars = ['<', '>', '"', "'"]
		if any(char in v for char in forbidden_chars):
			raise ValueError("Query contains forbidden characters")
		return v.strip()

	@field_validator("type")
	def validate_asset_type(cls, v):
		allowed_types = ["IMAGE", "VIDEO"]
		if v not in allowed_types:
			raise ValueError(f"Invalid asset type: {v}. Allowed values are {allowed_types}.")
		return v


class APIConfig(BaseModel):
	"""Configuration for Immich API connection."""
	base_url: HttpUrl = Field(os.getenv("IMMICH_URL"), description="Base URL for Immich API")
	api_key: str = Field(os.getenv("IMMICH_API_KEY"), min_length=1, description="API key for authentication")

	@field_validator("base_url")
	def validate_url(cls, v):
		if not v.endswith("/api"):
			raise ValueError("Base URL must end with '/api'.")
		return v


class ImageDimensions(BaseModel):
	"""Target dimensions for image processing."""
	width: int = Field(..., gt=0, description="Target width in pixels")
	height: int = Field(..., gt=0, description="Target height in pixels")

	@field_validator("width", "height")
	def validate_positive(cls, v):
		if v <= 0:
			raise ValueError("Width and height must be positive integers.")
		return v


class ProcessingConfig(BaseModel):
	"""Configuration for image processing operations."""
	target_dimensions: ImageDimensions = ImageDimensions(width=3840, height=2160)
	target_aspect_ratio: Tuple[int, int] = Field(default=(16, 9), description="Target aspect ratio as (width, height)")
	output_format: ImageFormat = ImageFormat.PNG
	output_filename: str = "current_image.png"

	@field_validator("output_filename")
	def validate_filename(cls, v):
		if not v.endswith((".png", ".jpeg", ".jpg", ".heic")):
			raise ValueError("Output filename must end with a supported file extension (.png, .jpeg, .heic).")
		return v.strip()

	@property
	def get_output_path(self) -> str:
		"""Generate full path for output file."""
		return IMAGES_DIR / self.output_filename


# Initialize Flask app and configs
app = Flask(__name__)
api_config = APIConfig()
processing_config = ProcessingConfig()


def error_handler(func):
	"""
	Decorator for consistent error handling across Flask routes.

	This decorator catches and properly formats various types of exceptions:
	- APIError: Network/API related issues (503 status)
	- ImageProcessingError: Image manipulation issues (500 status)
	- General exceptions: Unexpected errors (500 status)

	Args:
		func: The Flask route function to wrap

	Returns:
		Wrapped function with error handling
	"""

	@wraps(func)
	def wrapper(*args, **kwargs):
		try:
			return func(*args, **kwargs)
		except APIError as e:
			logger.error(f"API Error: {str(e)}")
			return jsonify({"status": "error", "message": str(e)}), 503
		except ImageProcessingError as e:
			logger.error(f"Image Processing Error: {str(e)}")
			return jsonify({"status": "error", "message": str(e)}), 500
		except Exception as e:
			logger.error(f"Unexpected Error: {str(e)}")
			return jsonify({"status": "error", "message": "Internal server error"}), 500

	return wrapper


def resize_to_4k(img: np.ndarray) -> np.ndarray:
	"""
	Resize an image to 4K resolution while maintaining aspect ratio.

	This function calculates the optimal scale factor to fit the image
	within 4K dimensions (3840x2160) without distortion.

	Args:
		img (np.ndarray): Input image as OpenCV array

	Returns:
		np.ndarray: Resized image array

	Raises:
		ImageProcessingError: If resize operation fails
	"""
	try:
		target = processing_config.target_dimensions
		h, w = img.shape[:2]
		scale = min(target.width / w, target.height / h)

		new_w = int(w * scale)
		new_h = int(h * scale)

		return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
	except Exception as e:
		raise ImageProcessingError(f"Failed to resize image: {str(e)}")


def get_salient_crop(img: np.ndarray, target_aspect_ratio: Optional[Tuple[int, int]] = None) -> np.ndarray:
	"""
	Crop image around the most salient region while maintaining specified aspect ratio.

	This function uses gradient-based saliency detection to identify the most
	interesting region of the image, then crops around that area while maintaining
	the target aspect ratio.

	Algorithm:
	1. Convert to grayscale and compute gradients using Sobel operators
	2. Calculate gradient magnitude as saliency measure
	3. Apply Gaussian blur to smooth the saliency map
	4. Find the point of maximum saliency
	5. Crop around this point with the target aspect ratio

	Args:
		img (np.ndarray): Input image as OpenCV array
		target_aspect_ratio (Optional[Tuple[int, int]]): Desired aspect ratio as (width, height).
			Defaults to config value if None.

	Returns:
		np.ndarray: Cropped image array

	Raises:
		ImageProcessingError: If crop operation fails
	"""
	try:
		target_aspect_ratio = target_aspect_ratio or processing_config.target_aspect_ratio
		height, width = img.shape[:2]

		# Convert to grayscale and calculate saliency
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
		grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
		gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
		gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
		saliency_map = cv2.GaussianBlur(gradient_magnitude, (21, 21), 0)

		_, _, _, max_loc = cv2.minMaxLoc(saliency_map)

		# Calculate crop dimensions
		crop_ratio = target_aspect_ratio[0] / target_aspect_ratio[1]
		crop_width = min(width, int(height * crop_ratio))
		crop_height = min(height, int(width / crop_ratio))

		x_center, y_center = max_loc
		x1 = max(0, x_center - crop_width // 2)
		y1 = max(0, y_center - crop_height // 2)
		x2 = min(width, x1 + crop_width)
		y2 = min(height, y1 + crop_height)

		# Adjust crop coordinates if necessary
		if x2 - x1 < crop_width:
			x1 = 0 if x1 == 0 else width - crop_width
			x2 = crop_width if x1 == 0 else width

		if y2 - y1 < crop_height:
			y1 = 0 if y1 == 0 else height - crop_height
			y2 = crop_height if y1 == 0 else height

		return img[y1:y2, x1:x2]
	except Exception as e:
		raise ImageProcessingError(f"Failed to crop image: {str(e)}")


def get_new_random_image(query: Optional[str] = None, range_limit: Optional[int] = None) -> bool:
	"""
	Fetch and process a random image from Immich API.

	This function performs the complete workflow:
	1. Search for images matching the query using cached requests
	2. Select a random image from search results
	3. Download the original image (bypasses cache)
	4. Fetch image metadata (cached)
	5. Process the image (crop and resize)
	6. Save the final image to the images directory

	The function uses smart caching to reduce API calls while ensuring
	fresh image data for downloads.

	Args:
		query (Optional[str]): Search query string. Defaults to "Mountains. Landscape. Nature"
		range_limit (Optional[int]): Maximum number of results to consider. Defaults to 50

	Returns:
		bool: True if image processing completed successfully

	Raises:
		APIError: If API requests fail or no images found
		ImageProcessingError: If image processing operations fail
	"""
	try:
		# Clear expired cache entries
		clear_expired_cache()

		# Use provided query or default
		search_query_text = query or os.getenv("DEFAULT_QUERY", "Mountains")
		result_limit = range_limit or 50

		# Prepare API request
		search_query = SearchQuery(
			query=search_query_text
		)

		headers = {
			'x-api-key': api_config.api_key,
			'Content-Type': 'application/json'
		}

		# Search for images - this will be cached
		response = cached_request(
			"POST",
			f"{api_config.base_url}/search/smart",
			headers=headers,
			json=search_query.model_dump()
		)
		response.raise_for_status()

		# Process response
		response_data = response.json()
		asset_count = min(result_limit, response_data["assets"]["count"])
		if asset_count == 0:
			raise APIError("No images found")

		random_asset_id = response_data["assets"]["items"][random.randint(0, asset_count - 1)]["id"]

		# Fetch image - this will NOT be cached (bypassed for /original endpoints)
		random_asset = cached_request(
			"GET",
			f"{api_config.base_url}/assets/{random_asset_id}/original",
			headers=headers
		)
		random_asset.raise_for_status()

		# Fetch metadata - this will be cached
		asset_metadata = cached_request(
			"GET",
			f"{api_config.base_url}/assets/{random_asset_id}",
			headers=headers
		)
		asset_metadata.raise_for_status()

		# Process image
		actual_extension = asset_metadata.json()["originalMimeType"].split("/")[1]
		temp_image_path = IMAGES_DIR / f"temp_random_asset.{actual_extension}"
		output_path = processing_config.get_output_path

		try:
			# Save temporary image file
			with open(temp_image_path, "wb") as f:
				f.write(random_asset.content)

			# Handle HEIC format
			if actual_extension.lower() == 'heic':
				from pillow_heif import register_heif_opener
				register_heif_opener()

			# Process the image
			with Image.open(temp_image_path) as img:
				# Convert to appropriate color mode
				if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
					img = img.convert('RGBA')
				else:
					img = img.convert('RGB')

				# Convert to OpenCV format and process
				opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
				cropped_img = get_salient_crop(opencv_img)
				resized_img = resize_to_4k(cropped_img)

				# Convert back to PIL and save final image
				final_img = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
				final_img.save(
					output_path,
					processing_config.output_format.value,
					optimize=True
				)

		finally:
			# Clean up temporary file
			if temp_image_path.exists():
				temp_image_path.unlink()

		logger.info(f"Successfully processed and saved image to: {output_path}")
		return True

	except requests.exceptions.RequestException as e:
		raise APIError(f"Failed to fetch image: {str(e)}")
	except Exception as e:
		raise ImageProcessingError(f"Failed to process image: {str(e)}")


@app.route('/new-image')
@error_handler
def new_image():
	"""
	Flask endpoint to generate a new processed image.

	This endpoint triggers the complete image processing workflow:
	fetching, processing, and saving a new random image.

	Query Parameters:
		query (str, optional): Search query for image selection
		range (int, optional): Maximum number of search results to consider (default: 50)

	Returns:
		JSON response indicating success or error status
	"""
	# Get optional query parameters
	query = request.args.get('query')
	range_limit = request.args.get('range', type=int)

	get_new_random_image(query=query, range_limit=range_limit)
	return jsonify({"status": "success", "message": "New image generated successfully"})


@app.route('/image')
@error_handler
def serve_image():
	"""
	Flask endpoint to serve the current processed image.

	Returns the most recently processed image file from the images directory.

	Returns:
		Flask file response with the current image

	Raises:
		ImageNotFoundError: If no processed image is available
	"""
	output_path = processing_config.get_output_path

	if not output_path.exists():
		raise ImageNotFoundError("No image available")

	return send_file(
		output_path,
		mimetype=f'image/{processing_config.output_format.value}'
	)


@app.route('/cache-status')
@error_handler
def cache_status():
	"""
	Flask endpoint to check cache status and statistics.

	Provides information about the current state of the API cache,
	including the number of entries and expiry configuration.

	Returns:
		JSON response with cache statistics
	"""
	clear_expired_cache()
	return jsonify({
		"status": "success",
		"cache_entries": len(_api_cache),
		"cache_expiry_hours": CACHE_EXPIRY_SECONDS / 3600,
		"images_directory": str(IMAGES_DIR.absolute())
	})


@app.route('/images-info')
@error_handler
def images_info():
	"""
	Flask endpoint to get information about the images directory.

	Returns statistics about saved images including directory path,
	current image status, and directory size.

	Returns:
		JSON response with images directory information
	"""
	output_path = processing_config.get_output_path

	return jsonify({
		"status": "success",
		"images_directory": str(IMAGES_DIR.absolute()),
		"current_image_exists": output_path.exists(),
		"current_image_path": str(output_path) if output_path.exists() else None,
		"current_image_size_bytes": output_path.stat().st_size if output_path.exists() else 0
	})


@app.route('/health', methods=['GET'])
@error_handler
def health_check():
	"""
	Health check endpoint for monitoring system status.

	Returns:
		JSON response with service status, container health, and dependency checks.
	"""

	try:
		# Check disk space in container
		total, used, free = shutil.disk_usage("/")
		disk_ok = free / total > 0.1  # Ensure at least 10% free disk space

		# Memory usage
		memory = psutil.virtual_memory()
		memory_ok = memory.available / memory.total > 0.1  # At least 10% memory free

		# Check Immich API connectivity
		try:
			response = requests.get(str(api_config.base_url))
			api_accessible = response.status_code == 404
		except ConnectionError:
			api_accessible = False

		# Determine overall status
		healthy = disk_ok and memory_ok and api_accessible

		# Construct the response
		return jsonify({
			"status": "healthy" if healthy else "unhealthy",
			"disk_usage": {
				"total_gb": total / (1024 ** 3),
				"used_gb": used / (1024 ** 3),
				"free_gb": free / (1024 ** 3),
				"status_ok": disk_ok
			},
			"memory_usage": {
				"total_gb": memory.total / (1024 ** 3),
				"available_gb": memory.available / (1024 ** 3),
				"status_ok": memory_ok
			},
			"api_connectivity": {
				"url": api_config.base_url,
				"status_ok": api_accessible
			}
		}), 200

	except Exception as e:
		logger.error(f"Health check error: {str(e)}")
		return jsonify({
			"status": "error",
			"message": str(e)
		}), 500


if __name__ == '__main__':
	# Ensure images directory exists and generate initial image
	IMAGES_DIR.mkdir(exist_ok=True)
	get_new_random_image()
	app.run()