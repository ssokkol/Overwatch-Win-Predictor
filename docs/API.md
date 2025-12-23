# API Documentation

## Base URL

```
http://localhost:8000
```

Production: `https://api.overwatch-predictor.com`

## Authentication

Most endpoints require an API key. Include it in the `X-API-Key` header:

```
X-API-Key: your-api-key-here
```

## Endpoints

### Health Check

Check API health status.

**Endpoint:** `GET /health`

**Authentication:** Not required

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### Predict Match Outcome

Predict the probability of victory for a match given two team compositions.

**Endpoint:** `POST /predict`

**Authentication:** Required

**Request Body:**
```json
{
  "team1": {
    "hero_ids": [1, 5, 10, 15, 20]
  },
  "team2": {
    "hero_ids": [2, 7, 12, 17, 22]
  }
}
```

**Validation:**
- Each team must have exactly 5 heroes
- Hero IDs must be between 1 and 38
- No duplicate heroes within a team
- No heroes can appear in both teams

**Response:**
```json
{
  "team1_win_probability": 0.65,
  "team2_win_probability": 0.35,
  "predicted_winner": "Team1",
  "confidence": 0.65
}
```

**Error Responses:**

- `400 Bad Request`: Invalid request format
- `401 Unauthorized`: Missing or invalid API key
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

### Get Hero Recommendations

Get recommended hero picks based on current and enemy team compositions.

**Endpoint:** `POST /recommendations`

**Authentication:** Required

**Request Body:**
```json
{
  "current_team": {
    "hero_ids": [1, 5, 10, 15, 20]
  },
  "enemy_team": {
    "hero_ids": [2, 7, 12, 17, 22]
  },
  "num_recommendations": 5
}
```

**Response:**
```json
{
  "recommended_heroes": [3, 8, 11, 16, 21],
  "reasoning": "Recommended heroes to balance team composition and provide strong counter-picks."
}
```

## Rate Limiting

- Default: 100 requests per 60 seconds per IP address
- Rate limit information is provided in response headers:
  - `X-RateLimit-Remaining`: Number of requests remaining

## Error Handling

All errors follow this format:

```json
{
  "detail": "Error message description"
}
```

## Example Usage

### Python

```python
import requests

API_KEY = "your-api-key"
BASE_URL = "http://localhost:8000"

# Predict match
response = requests.post(
    f"{BASE_URL}/predict",
    headers={"X-API-Key": API_KEY},
    json={
        "team1": {"hero_ids": [1, 5, 10, 15, 20]},
        "team2": {"hero_ids": [2, 7, 12, 17, 22]}
    }
)
print(response.json())
```

### cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "team1": {"hero_ids": [1, 5, 10, 15, 20]},
    "team2": {"hero_ids": [2, 7, 12, 17, 22]}
  }'
```

