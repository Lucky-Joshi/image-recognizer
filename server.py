from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import base64
from PIL import Image
import io
import json
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Gemini API - Replace with your actual API key
GEMINI_API_KEY = 'Your-API-Key'
genai.configure(api_key=GEMINI_API_KEY)

@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'Object Recognition API is running',
        'version': '1.0'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """
    API endpoint to analyze image with Gemini
    
    Request JSON:
    {
        "image": "base64_encoded_image_data",
        "object_name": "object_to_find"
    }
    
    Response JSON:
    {
        "success": true,
        "result": {
            "found": true/false,
            "confidence": "high/medium/low",
            "description": "description",
            "location": "location info",
            "additional_objects": ["object1", "object2"]
        }
    }
    """
    try:
        # Get request data
        data = request.json
        image_data = data.get('image')
        object_name = data.get('object_name')

        # Validation
        if not image_data or not object_name:
            return jsonify({
                'error': 'Missing required fields: image and object_name',
                'success': False
            }), 400

        # Remove data URL prefix if present (data:image/png;base64,)
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({
                'error': 'Invalid base64 image data',
                'success': False
            }), 400

        # Open image with PIL
        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({
                'error': 'Invalid image format',
                'success': False
            }), 400

        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.5-flash')

        # Create detailed prompt
        prompt = f"""Analyze this image carefully and determine if there is a "{object_name}" present in the image.

Respond ONLY in valid JSON format with the following structure (no additional text before or after):
{{
    "found": true or false,
    "confidence": "high" or "medium" or "low",
    "description": "brief description of what you see related to the object or why it wasn't found",
    "location": "specific location in the image where the object is found, or 'Not applicable' if not found",
    "additional_objects": ["list", "of", "other", "notable", "objects"]
}}

Be precise and accurate in your analysis. Only set "found" to true if you are confident the {object_name} is actually present in the image."""

        # Generate response from Gemini
        response = model.generate_content([prompt, image])
        result_text = response.text

        # Parse JSON from response
        json_match = re.search(r'\{[\s\S]*\}', result_text)
        
        if json_match:
            try:
                result = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                # If JSON parsing fails, create fallback result
                result = create_fallback_result(result_text, object_name)
        else:
            # No JSON found, create fallback result
            result = create_fallback_result(result_text, object_name)

        # Validate result structure
        result = validate_result(result)

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        print(f"Error in analyze_image: {str(e)}")
        return jsonify({
            'error': f'Server error: {str(e)}',
            'success': False
        }), 500

def create_fallback_result(text, object_name):
    """Create a fallback result when JSON parsing fails"""
    found_keywords = ['yes', 'found', 'present', 'visible', 'see', 'true']
    not_found_keywords = ['no', 'not found', 'absent', 'cannot', 'false']
    
    text_lower = text.lower()
    
    # Determine if object was found
    found = False
    for keyword in found_keywords:
        if keyword in text_lower:
            found = True
            break
    
    for keyword in not_found_keywords:
        if keyword in text_lower:
            found = False
            break
    
    return {
        'found': found,
        'confidence': 'medium',
        'description': text[:500],  # Limit description length
        'location': 'See description' if found else 'Not applicable',
        'additional_objects': []
    }

def validate_result(result):
    """Validate and ensure result has all required fields"""
    validated = {
        'found': bool(result.get('found', False)),
        'confidence': result.get('confidence', 'medium'),
        'description': result.get('description', 'No description available'),
        'location': result.get('location', 'Unknown'),
        'additional_objects': result.get('additional_objects', [])
    }
    
    # Ensure confidence is valid
    if validated['confidence'] not in ['high', 'medium', 'low']:
        validated['confidence'] = 'medium'
    
    # Ensure additional_objects is a list
    if not isinstance(validated['additional_objects'], list):
        validated['additional_objects'] = []
    
    return validated

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    api_configured = GEMINI_API_KEY != 'Your-API-Key'
    
    return jsonify({
        'status': 'healthy',
        'api_configured': api_configured,
        'model': 'gemini-1.5-flash'
    })

if __name__ == '__main__':
    # Check if API key is configured
    if GEMINI_API_KEY == 'Your-API-Key':
        print("\n" + "="*60)
        print("⚠️  WARNING: Gemini API key not configured!")
        print("="*60)
        print("Please set your API key in the code:")
        print("GEMINI_API_KEY = 'AIzaSyCj7m59ncV1FDh6xIH3mzfu8ajw2ezYpgI'")
        print("\nGet your API key from:")
        print("https://makersuite.google.com/app/apikey")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("✅ API Key configured successfully!")
        print("="*60 + "\n")
    
    print("🚀 Starting Object Recognition Server...")
    print("📍 Server running at: http://localhost:5000")
    print("🔗 API endpoint: http://localhost:5000/api/analyze")
    print("\n💡 Press CTRL+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
