import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="Gemini Multimodal Embedding Microservice")

import os

# Initialize the Gemini client. 
# We default client to None so we can initialize it lazily in the route
client = None
try:
    # If the environment signals Vertex AI (like in Cloud Run), we initialize using Vertex AI natively.
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true" or "GOOGLE_CLOUD_PROJECT" in os.environ:
        client = genai.Client(vertexai=True)
    else:
        client = genai.Client()
except Exception as e:
    print(f"Warning: Could not initialize genai.Client at startup. Error: {e}")

class EmbeddingRequest(BaseModel):
    image_base64: str

class EmbeddingResponse(BaseModel):
    vector: list[float]

@app.post("/embed_image", response_model=EmbeddingResponse)
async def embed_image(request: EmbeddingRequest):
    try:
        # Decode the Base64 image
        # In case the client sends data URI format (e.g., 'data:image/png;base64,iVBORw0KGgo...')
        base64_data = request.image_base64
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
            
        try:
            image_bytes = base64.b64decode(base64_data)
        except Exception as decode_error:
            raise HTTPException(status_code=400, detail=f"Invalid Base64 string: {decode_error}")
        
        global client
        if client is None:
            try:
                if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true" or "GOOGLE_CLOUD_PROJECT" in os.environ:
                    client = genai.Client(
            vertexai=True, 
            project="sudhahar-image-compare", 
            location="us-central1" # Ensure this matches your deployment region
                    )
                else:
                    client = genai.Client()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Gemini Client not initialized and failed to start: {e}")

        # Create the image part for Gemini Embedding
        # The genai SDK supports bytes with mime type 
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/png', # Setting to png; Gemini usually parses standard formats transparently
        )
        
        # Generate the embedding
        result = client.models.embed_content(
            model='gemini-embedding-2',
            contents=[image_part],
            #contents=[
             #   types.Part.from_bytes(
              #      data=image_bytes,
               #     mime_type='image/png' # For OWS, consider 'image/jpeg' if that's your standard
                #)
            #],
            
            config=types.EmbedContentConfig(
                output_dimensionality=1408
            )
        )
        
        if result.embeddings and len(result.embeddings) > 0:
            vector = result.embeddings[0].values
            return {"vector": vector}
        else:
            raise HTTPException(status_code=500, detail="No embeddings returned from the model.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
