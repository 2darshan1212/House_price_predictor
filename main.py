from pathlib import Path
import logging
import math

import joblib
import pandas as pd

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_predict.joblib"
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EstateAI - House Price Predictor",
    description="California house price prediction using machine learning.",
    version="1.0.0",
)

# Mount static files using an absolute path.
app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


# ==========================================
# 2. LOAD THE TRAINED MODEL
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

artifact = joblib.load(MODEL_PATH)

# Your saved artifact is a dictionary containing
# the trained pipeline and feature list.
if isinstance(artifact, dict):
    model = artifact["model"]
    FEATURES = artifact.get(
        "feature",
        [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
            "ocean_proximity",
        ],
    )
else:
    # Fallback for an artifact containing only a model.
    model = artifact
    FEATURES = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "ocean_proximity",
    ]

logger.info("House price model loaded successfully.")
logger.info("Model features: %s", FEATURES)


# ==========================================
# 3. INPUT SCHEMA FOR JSON API
# ==========================================

class HouseFeatures(BaseModel):
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)
    housing_median_age: float = Field(ge=0)
    total_rooms: float = Field(ge=0)
    total_bedrooms: float = Field(ge=0)
    population: float = Field(ge=0)
    households: float = Field(ge=0)
    median_income: float = Field(ge=0)
    ocean_proximity: str


ALLOWED_PROXIMITIES = {
    "INLAND",
    "NEAR BAY",
    "NEAR OCEAN",
    "ISLAND",
    "<1H OCEAN",
}


# ==========================================
# 4. SHARED PREDICTION FUNCTION
# ==========================================

def predict_house_price(values: dict) -> float:
    """Validate the features and return a model prediction."""

    proximity = str(values["ocean_proximity"]).strip().upper()

    if proximity not in ALLOWED_PROXIMITIES:
        raise ValueError("Invalid ocean proximity selected.")

    values["ocean_proximity"] = proximity

    numeric_features = [
        feature
        for feature in FEATURES
        if feature != "ocean_proximity"
    ]

    for feature in numeric_features:
        number = float(values[feature])

        if not math.isfinite(number):
            raise ValueError(
                f"{feature} must be a valid finite number."
            )

        if number < 0 and feature not in {
            "longitude",
            "latitude",
        }:
            raise ValueError(
                f"{feature} cannot be negative."
            )

        values[feature] = number

    # Preserve the feature order used during training.
    input_df = pd.DataFrame(
        [[values[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    result = model.predict(input_df)
    prediction = float(result[0])

    if not math.isfinite(prediction):
        raise ValueError(
            "The model returned an invalid prediction."
        )

    return prediction


# ==========================================
# 5. HOME PAGE
# ==========================================

@app.get("/", response_class=HTMLResponse, name="home")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": None,
            "error": None,
            "form_data": None,
        },
    )


# ==========================================
# 6. HTML FORM PREDICTION
# ==========================================

@app.post("/predict", response_class=HTMLResponse)
async def predict_from_form(request: Request):
    form = await request.form()

    form_data = {
        feature: str(form.get(feature, "")).strip()
        for feature in FEATURES
    }

    try:
        values = {}

        for feature in FEATURES:
            raw_value = form_data[feature]

            if not raw_value:
                raise ValueError(
                    f"Please provide a value for {feature}."
                )

            if feature == "ocean_proximity":
                values[feature] = raw_value
            else:
                values[feature] = float(raw_value)

        prediction = predict_house_price(values)

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "prediction": prediction,
                "error": None,
                "form_data": form_data,
            },
        )

    except (ValueError, TypeError) as exc:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "prediction": None,
                "error": str(exc),
                "form_data": form_data,
            },
            status_code=400,
        )

    except Exception:
        logger.exception("HTML prediction failed.")

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "prediction": None,
                "error": (
                    "Prediction failed. Check the server terminal "
                    "for the technical details."
                ),
                "form_data": form_data,
            },
            status_code=500,
        )


# ==========================================
# 7. JSON API
# ==========================================

@app.post("/api/predict")
async def predict_from_api(data: HouseFeatures):
    values = data.model_dump()

    try:
        prediction = predict_house_price(values)

        return {
            "success": True,
            "predicted_price": round(prediction, 2),
            "currency": "USD",
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("API prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check server logs.",
        ) from exc


# ==========================================
# 8. HEALTH CHECK
# ==========================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


# ==========================================
# 9. STARTUP MESSAGE
# ==========================================

@app.on_event("startup")
async def startup_message():
    logger.info("EstateAI server is ready.")