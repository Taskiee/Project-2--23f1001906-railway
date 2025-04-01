#Upload your losslessly compressed image (less than 1,500 bytes)


from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import io
import os
import asyncio
from pathlib import Path
import tempfile
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def compress_losslessly(
    image_data: bytes,
    max_size: int = 1500,
    max_attempts: int = 5
) -> Tuple[bool, bytes, int]:
    """Losslessly compress an image to under target size.
    
    Args:
        image_data: Raw image bytes
        max_size: Target max size in bytes
        max_attempts: Maximum compression attempts
        
    Returns:
        Tuple of (success: bool, compressed_data: bytes, final_size: int)
    """
    try:
        with Image.open(io.BytesIO(image_data)) as img:
            best_size = float('inf')
            best_buffer = None
            
            # Try multiple compression methods
            for attempt in range(1, max_attempts + 1):
                buffer = io.BytesIO()
                params = {
                    'format': 'WEBP',
                    'lossless': True,
                    'method': 6 - min(attempt, 3),  # Max method=6
                    'quality': 100  # Maintain lossless
                }
                
                img.save(buffer, **params)
                current_size = buffer.tell()
                
                if current_size < best_size:
                    best_size = current_size
                    best_buffer = buffer
                
                if current_size <= max_size:
                    break
            
            if best_buffer:
                return (best_size <= max_size, best_buffer.getvalue(), best_size)
            
            return (False, None, 0)
            
    except Exception as e:
        logger.error(f"Compression failed: {str(e)}")
        raise

@app.post("/compress/")
async def compress_image(file: UploadFile = File(...)):
    """API endpoint for image compression"""
    try:
        # Read uploaded file
        image_data = await file.read()
        original_size = len(image_data)
        
        # Compress image
        success, compressed_data, final_size = await compress_losslessly(image_data)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Could not compress under 1500 bytes (best: {final_size} bytes)"
            )
        
        # Save to temp file
        temp_dir = tempfile.mkdtemp()
        output_path = Path(temp_dir) / "compressed.webp"
        with open(output_path, 'wb') as f:
            f.write(compressed_data)
        
        logger.info(
            f"Compressed {file.filename}: {original_size} → {final_size} bytes"
        )
        
        return FileResponse(
            output_path,
            media_type="image/webp",
            headers={
                "Original-Size": str(original_size),
                "Compressed-Size": str(final_size)
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)