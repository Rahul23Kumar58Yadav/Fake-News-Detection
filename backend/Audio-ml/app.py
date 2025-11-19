from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import re
import os
import torch
from transformers import pipeline
from datetime import datetime
import traceback

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_DIR = "models"
BERT_MODEL_NAME = "facebook/bart-large-mnli"

# Create models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Global variables for ML models
vectorization = None
LR = None
gbc = None
rfc = None
zero_shot_classifier = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for history storage (IN-MEMORY)
predictions_history = []
prediction_counter = 0

# ---------------------------------------------------------------
# Load Zero-Shot BERT Model
# ---------------------------------------------------------------
def load_bert_model():
    """Load zero-shot classification model"""
    global zero_shot_classifier
    try:
        print(f"Loading zero-shot BERT model: {BERT_MODEL_NAME}")
        zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model=BERT_MODEL_NAME,
            device=0 if torch.cuda.is_available() else -1
        )
        print("✓ Zero-shot BERT model loaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Error loading BERT model: {e}")
        return False


# ---------------------------------------------------------------
# Load Traditional ML Models (optional)
# ---------------------------------------------------------------
def load_traditional_models():
    """Load traditional ML models if available"""
    global vectorization, LR, gbc, rfc
    try:
        if os.path.exists(os.path.join(MODEL_DIR, 'vectorizer.pkl')):
            vectorization = joblib.load(os.path.join(MODEL_DIR, 'vectorizer.pkl'))
            LR = joblib.load(os.path.join(MODEL_DIR, 'lr_model.pkl'))
            gbc = joblib.load(os.path.join(MODEL_DIR, 'gbc_model.pkl'))
            rfc = joblib.load(os.path.join(MODEL_DIR, 'rfc_model.pkl'))
            print("✓ Traditional ML models loaded!")
            return True
    except Exception as e:
        print(f"⚠ Traditional models not available: {e}")
        return False


# ---------------------------------------------------------------
# Initialize Models on Startup
# ---------------------------------------------------------------
print("=" * 50)
print("Initializing Fake News Detection System")
print("=" * 50)
bert_available = load_bert_model()
traditional_available = load_traditional_models()

if not bert_available and not traditional_available:
    print("❌ ERROR: No models available!")
else:
    print("✓ System ready!")
print("=" * 50)


# ---------------------------------------------------------------
# Text Preprocessing
# ---------------------------------------------------------------
def preprocess_text(text):
    """Clean and preprocess text"""
    try:
        if not text or not isinstance(text, str):
            return ""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.lower()
        text = ' '.join(text.split())
        return text
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return ""


# ---------------------------------------------------------------
# Zero-Shot Fake News Prediction
# ---------------------------------------------------------------
def predict_with_bert(text):
    """Improved zero-shot fake news detection using strong hypothesis contrast"""
    try:
        if zero_shot_classifier is None:
            print("BERT model not available")
            return None
            
        # Truncate text if too long (BERT has token limits)
        max_length = 512
        if len(text.split()) > max_length:
            text = ' '.join(text.split()[:max_length])
            
        candidate_labels = ["fake", "real"]
        hypothesis_template = "This statement is {} news."

        result = zero_shot_classifier(
            sequences=text,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
            multi_label=False
        )

        # Extract top label and probabilities
        scores = dict(zip(result["labels"], result["scores"]))
        fake_score = scores.get("fake", 0) * 100
        real_score = scores.get("real", 0) * 100

        label = "Fake" if fake_score > real_score else "Real"
        confidence = max(fake_score, real_score)

        return {
            "prediction": label,
            "confidence": round(confidence, 2),
            "probabilities": {
                "fake": round(fake_score, 2),
                "real": round(real_score, 2)
            }
        }

    except Exception as e:
        print(f"BERT zero-shot prediction error: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------
# Traditional ML Prediction (Fallback)
# ---------------------------------------------------------------
def predict_with_traditional(text):
    """Predict using traditional ML models"""
    try:
        if not all([vectorization, LR, gbc, rfc]):
            return None

        processed_text = preprocess_text(text)
        if not processed_text:
            return None
            
        df = pd.DataFrame({"text": [processed_text]})
        X = vectorization.transform(df["text"])

        pred_lr = LR.predict(X)[0]
        pred_gbc = gbc.predict(X)[0]
        pred_rfc = rfc.predict(X)[0]

        prob_lr = LR.predict_proba(X)[0]
        prob_gbc = gbc.predict_proba(X)[0]
        prob_rfc = rfc.predict_proba(X)[0]

        return {
            "LR": {"prediction": "Fake" if pred_lr == 0 else "Real", "confidence": round(max(prob_lr) * 100, 2)},
            "GBC": {"prediction": "Fake" if pred_gbc == 0 else "Real", "confidence": round(max(prob_gbc) * 100, 2)},
            "RFC": {"prediction": "Fake" if pred_rfc == 0 else "Real", "confidence": round(max(prob_rfc) * 100, 2)},
        }

    except Exception as e:
        print(f"Traditional prediction error: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------
# Main Prediction Endpoint
# ---------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    """Main prediction endpoint for frontend/backend"""
    global prediction_counter, predictions_history
    
    print("\n" + "="*50)
    print("PREDICT ENDPOINT CALLED")
    print("="*50)
    
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
            
        print(f"Received data type: {type(data)}")
        print(f"Received data: {data}")
        
        if not data:
            print("ERROR: No data received")
            return jsonify({"success": False, "error": "No data received"}), 400
            
        text = data.get('text', '').strip()
        print(f"Text length: {len(text)} characters")

        # Validation
        if not text:
            return jsonify({"success": False, "error": "Text is required"}), 400
        if len(text) < 20:
            return jsonify({"success": False, "error": "Text too short (minimum 20 characters)"}), 400
        if len(text) > 10000:
            return jsonify({"success": False, "error": "Text too long (maximum 10000 characters)"}), 400

        print(f"Processing text: {text[:100]}...")
        
        results = {}
        
        # Try BERT first
        if bert_available and zero_shot_classifier is not None:
            print("Attempting BERT prediction...")
            bert_result = predict_with_bert(text)
            if bert_result:
                results["BERT"] = bert_result
                print(f"✓ BERT prediction: {bert_result}")

        # Try traditional models
        if traditional_available:
            print("Attempting traditional ML prediction...")
            trad_result = predict_with_traditional(text)
            if trad_result:
                results.update(trad_result)
                print(f"✓ Traditional predictions: {list(trad_result.keys())}")

        print(f"\nTotal results: {list(results.keys())}")

        # If no models produced results
        if not results:
            return jsonify({
                "success": False,
                "error": "Models failed to generate predictions",
                "bert_available": bert_available,
                "traditional_available": traditional_available,
                "suggestion": "Check if models are properly loaded"
            }), 500

        # Build response with proper fallbacks
        def get_value(results, keys, key, default):
            for k in keys:
                if k in results and key in results[k]:
                    val = results[k][key]
                    if val is not None:
                        return val
            return default
        
        lr_pred = get_value(results, ["LR", "BERT"], "prediction", "Unknown")
        gbc_pred = get_value(results, ["GBC", "BERT"], "prediction", "Unknown")
        rfc_pred = get_value(results, ["RFC", "BERT"], "prediction", "Unknown")
        
        lr_conf = get_value(results, ["LR", "BERT"], "confidence", 0.0)
        gbc_conf = get_value(results, ["GBC", "BERT"], "confidence", 0.0)
        rfc_conf = get_value(results, ["RFC", "BERT"], "confidence", 0.0)

        # Store in history
        prediction_counter += 1
        history_entry = {
            "_id": f"pred_{prediction_counter}",
            "text": text[:200] + "..." if len(text) > 200 else text,
            "predictions": {
                "lr": lr_pred,
                "gbc": gbc_pred,
                "rfc": rfc_pred
            },
            "confidence": {
                "lr": lr_conf,
                "gbc": gbc_conf,
                "rfc": rfc_conf
            },
            "timestamp": datetime.now().isoformat()
        }
        predictions_history.insert(0, history_entry)
        
        # Keep only last 100 predictions
        if len(predictions_history) > 100:
            predictions_history.pop()

        response = {
            "success": True,
            "predictions": {
                "logisticRegression": lr_pred,
                "gradientBoosting": gbc_pred,
                "randomForest": rfc_pred,
            },
            "confidence": {
                "logisticRegression": float(lr_conf),
                "gradientBoosting": float(gbc_conf),
                "randomForest": float(rfc_conf),
            },
            "bert": results.get("BERT"),
            "processedText": preprocess_text(text),
            "models_used": {
                "bert": "BERT" in results,
                "traditional": any(k in results for k in ["LR", "GBC", "RFC"])
            },
            "id": history_entry["_id"]
        }

        print(f"✓ Response: {response['predictions']}")
        print("="*50 + "\n")
        
        return jsonify(response), 200

    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        
        print(f"\n✗ EXCEPTION: {error_type}")
        print(f"  Message: {error_msg}")
        traceback.print_exc()
        print("="*50 + "\n")
        
        return jsonify({
            "success": False,
            "error": error_msg,
            "type": error_type,
            "suggestion": "Check server logs for details"
        }), 500


# ---------------------------------------------------------------
# History Endpoint
# ---------------------------------------------------------------
@app.route("/api/history", methods=["GET"])
def get_history():
    """Get prediction history"""
    try:
        limit = min(int(request.args.get('limit', 20)), 100)
        page = max(int(request.args.get('page', 1)), 1)
        
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        
        paginated_history = predictions_history[start_idx:end_idx]
        
        return jsonify({
            "success": True,
            "data": paginated_history,
            "pagination": {
                "currentPage": page,
                "totalPages": max(1, (len(predictions_history) + limit - 1) // limit),
                "totalItems": len(predictions_history),
                "itemsPerPage": limit
            }
        }), 200
    except Exception as e:
        print(f"History error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------
# Statistics Endpoint
# ---------------------------------------------------------------
@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get prediction statistics"""
    try:
        if not predictions_history:
            return jsonify({
                "success": True,
                "stats": {
                    "total": 0,
                    "fake": 0,
                    "real": 0,
                    "fakePercentage": 0,
                    "realPercentage": 0
                }
            }), 200
        
        total = len(predictions_history)
        fake_count = sum(1 for p in predictions_history 
                        if p.get('predictions', {}).get('lr') == 'Fake' or
                           p.get('predictions', {}).get('gbc') == 'Fake' or
                           p.get('predictions', {}).get('rfc') == 'Fake')
        
        real_count = total - fake_count
        
        return jsonify({
            "success": True,
            "stats": {
                "total": total,
                "fake": fake_count,
                "real": real_count,
                "fakePercentage": round((fake_count / total) * 100, 2),
                "realPercentage": round((real_count / total) * 100, 2)
            }
        }), 200
    except Exception as e:
        print(f"Stats error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "OK",
        "models": {
            "bert": bert_available,
            "traditional_ml": traditional_available
        },
        "device": str(device),
        "message": "ML API is running"
    }), 200


# ---------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"success": False, "error": "Internal server error"}), 500


# ---------------------------------------------------------------
# Start Server
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 FAKE NEWS DETECTION API")
    print("="*60)
    print(f"✅ BERT: {'Loaded' if bert_available else 'Not Available'}")
    print(f"✅ Traditional ML: {'Loaded' if traditional_available else 'Not Available'}")
    print("="*60)
    print("📍 Endpoints:")
    print("   GET  /api/health")
    print("   POST /api/predict")
    print("   GET  /api/history")
    print("   GET  /api/stats")
    print("="*60)
    print("🌐 Server: http://localhost:5001")
    print("="*60 + "\n")
    
    app.run(host="0.0.0.0", port=5001, debug=True)