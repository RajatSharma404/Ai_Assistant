"""
Quick Test: Start Learning API Server with static file serving
"""
import sys
import os
os.chdir('f:/bn/assitant')
sys.path.insert(0, 'f:/bn/assitant')

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    from ai_assistant.services.learning_api import router
    
    app = FastAPI(title="Learning Systems API")
    app.include_router(router)
    
    # Serve static files
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    # Dashboard route
    @app.get("/")
    async def dashboard():
        return FileResponse("static/learning_dashboard.html")
    
    print("🚀 Starting server on http://127.0.0.1:8000")
    print("📊 Dashboard at http://127.0.0.1:8000/")
    print("📚 API Docs at http://127.0.0.1:8000/docs")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
