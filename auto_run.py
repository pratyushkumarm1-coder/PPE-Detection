import sys
import subprocess

print("Installing python-multipart...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "python-multipart"])

print("Starting the server...")
import uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
