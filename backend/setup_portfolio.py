#!/usr/bin/env python3
"""
Smart setup script untuk AI Portfolio Backend
Handles Windows development + Railway production deployment
"""

import sys
import subprocess
import os
import json
import shutil

def print_header(title):
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def print_step(step, description):
    print(f"\n📋 Step {step}: {description}")
    print("-" * 40)

def run_command(cmd, description, critical=True):
    """run command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED")
            if critical:
                print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {description} - EXCEPTION: {e}")
        return False

def check_python_version():
    """check python version"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    if version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print("✅ Python version compatible")
    return True

def check_required_files():
    """check if required files exist"""
    required_files = {
        "portfolio.json": "Knowledge base dengan informasi portfolio",
        "simple_rag_system.py": "Simple RAG system untuk Windows fallback",
        "requirements.txt": "Dependencies list untuk production"
    }
    
    missing_files = []
    for file, desc in required_files.items():
        if os.path.exists(file):
            print(f"✅ {file} - Found")
        else:
            print(f"❌ {file} - Missing ({desc})")
            missing_files.append(file)
    
    return len(missing_files) == 0, missing_files

def validate_portfolio_json():
    """validate portfolio.json format"""
    try:
        with open("portfolio.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("❌ portfolio.json should be a list")
            return False
        
        required_fields = ["id", "category", "title", "content"]
        for i, doc in enumerate(data):
            for field in required_fields:
                if field not in doc:
                    print(f"❌ Document {i} missing field: {field}")
                    return False
        
        print(f"✅ portfolio.json valid with {len(data)} documents")
        categories = set(doc["category"] for doc in data)
        print(f"   Categories: {', '.join(sorted(categories))}")
        return True
        
    except Exception as e:
        print(f"❌ Error validating portfolio.json: {e}")
        return False

def setup_environment():
    """setup .env file"""
    if os.path.exists(".env"):
        print("✅ .env file already exists")
        return True
    
    if os.path.exists(".env.example"):
        shutil.copy(".env.example", ".env")
        print("✅ .env file created from template")
        print("⚠️  EDIT .env file and add your OpenAI API key!")
        return True
    else:
        # create basic .env
        with open(".env", "w") as f:
            f.write("# OpenAI API Configuration\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n\n")
            f.write("# Frontend URL Configuration\n")
            f.write("FRONTEND_URL=https://frontend-portofolio-danen.vercel.app\n")
        
        print("✅ Basic .env file created")
        print("⚠️  EDIT .env file and add your OpenAI API key!")
        return True

def install_core_dependencies():
    """install core dependencies yang pasti bisa di-install"""
    print_step("1", "Installing Core Dependencies")
    
    core_packages = [
        "fastapi",
        "uvicorn", 
        "pydantic",
        "python-dotenv",
        "requests",
        "openai"
    ]
    
    # update pip dulu
    if not run_command(f"{sys.executable} -m pip install --upgrade pip setuptools wheel", "Updating pip and setuptools"):
        return False
    
    # install core packages
    for package in core_packages:
        if not run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}"):
            print(f"❌ Failed to install {package}")
            return False
    
    # test core imports
    try:
        import fastapi, uvicorn, openai
        print("✅ Core packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Core import failed: {e}")
        return False

def install_ml_dependencies():
    """install ML dependencies dengan fallback handling"""
    print_step("2", "Installing ML Dependencies")
    
    ml_packages = [
        "numpy",
        "scikit-learn", 
        "sentence-transformers"
    ]
    
    failed_packages = []
    
    for package in ml_packages:
        if run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}", critical=False):
            print(f"✅ {package} installed successfully")
        else:
            print(f"❌ {package} installation failed")
            failed_packages.append(package)
    
    # test ML imports
    try:
        import numpy, sklearn, sentence_transformers
        print("✅ All ML packages working")
        return True, []
    except ImportError as e:
        print(f"⚠️  Some ML packages missing: {e}")
        return False, failed_packages

def install_chromadb():
    """coba install chromadb - ok kalau gagal"""
    print_step("3", "Installing ChromaDB (Optional)")
    
    print("🔧 Attempting to install ChromaDB...")
    print("   (This might fail on Windows - that's OK!)")
    
    if run_command(f"{sys.executable} -m pip install chromadb", "Installing ChromaDB", critical=False):
        try:
            import chromadb
            print("✅ ChromaDB installed and working!")
            return True
        except ImportError:
            print("❌ ChromaDB installed but not importable")
            return False
    else:
        print("❌ ChromaDB installation failed (expected on Windows)")
        print("✅ Will use Simple RAG system as fallback")
        return False

def test_rag_systems():
    """test which rag systems are available"""
    print_step("4", "Testing RAG Systems")
    
    chromadb_available = False
    simple_rag_available = False
    
    # test chromadb
    try:
        import chromadb
        from rag_system import RAGSystem
        print("✅ ChromaDB RAG system available")
        chromadb_available = True
    except ImportError:
        print("❌ ChromaDB RAG system not available")
    
    # test simple rag
    try:
        from simple_rag_system import SimpleRAGSystem
        print("✅ Simple RAG system available")
        simple_rag_available = True
    except ImportError:
        print("❌ Simple RAG system not available")
    
    return chromadb_available, simple_rag_available

def test_complete_system():
    """test complete system dengan portfolio AI"""
    print_step("5", "Testing Complete System")
    
    # test main.py import
    try:
        # add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # test import main module
        import main
        print("✅ main.py imports successfully")
        
        # test portfolio loading
        knowledge_data = main.load_portfolio_knowledge()
        if knowledge_data:
            print(f"✅ Portfolio knowledge loaded ({len(knowledge_data)} documents)")
        else:
            print("❌ Portfolio knowledge loading failed")
            return False
        
        # test rag initialization
        if main.initialize_rag_system:
            rag = main.initialize_rag_system(knowledge_data, use_openai=False)
            if rag:
                print(f"✅ RAG system initialized ({main.RAG_SYSTEM_TYPE})")
            else:
                print("❌ RAG system initialization failed")
                return False
        else:
            print("⚠️  RAG system not available - will use fallback responses")
        
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def create_test_server():
    """create simple test server file"""
    test_server_code = '''from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Portfolio AI Test Server")

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {
        "message": "Portfolio AI Test Server Running!", 
        "status": "test_mode",
        "instructions": "Try POST /ask with question"
    }

@app.post("/ask")
def ask_test(request: QuestionRequest):
    return {
        "response": f"Test response for: {request.question}",
        "session_id": "test_session"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test_server:app", host="0.0.0.0", port=8000, reload=True)
'''
    
    with open("test_server.py", "w") as f:
        f.write(test_server_code)
    
    print("✅ test_server.py created")

def main():
    print_header("AI Portfolio Backend Setup")
    print("Smart setup untuk Windows development + Railway deployment")
    
    # prerequisite checks
    if not check_python_version():
        print("\n❌ Setup failed: Python version incompatible")
        sys.exit(1)
    
    files_ok, missing = check_required_files()
    if not files_ok:
        print(f"\n❌ Setup failed: Missing files - {', '.join(missing)}")
        print("\n📝 Required files:")
        print("   - portfolio.json (your knowledge base)")
        print("   - simple_rag_system.py (from artifacts)")
        print("   - requirements.txt (from artifacts)")
        sys.exit(1)
    
    if not validate_portfolio_json():
        print("\n❌ Setup failed: Invalid portfolio.json")
        sys.exit(1)
    
    setup_environment()
    
    # installation steps
    if not install_core_dependencies():
        print("\n❌ Setup failed: Core dependencies installation failed")
        sys.exit(1)
    
    ml_success, ml_failed = install_ml_dependencies()
    if not ml_success:
        print(f"\n⚠️  ML dependencies partially failed: {ml_failed}")
        print("Will continue with available packages...")
    
    chromadb_success = install_chromadb()
    
    # test systems
    chromadb_available, simple_available = test_rag_systems()
    
    if not chromadb_available and not simple_available:
        print("\n❌ No RAG system available!")
        print("Portfolio will work with basic fallback responses only.")
    
    # test complete system
    system_test_ok = test_complete_system()
    
    # create test server
    create_test_server()
    
    # final results
    print_header("SETUP RESULTS")
    
    print("📊 System Status:")
    print(f"   Core Dependencies: {'✅ Working' if True else '❌ Failed'}")
    print(f"   ML Dependencies: {'✅ Working' if ml_success else '⚠️ Partial'}")
    print(f"   ChromaDB: {'✅ Available' if chromadb_available else '❌ Not Available'}")
    print(f"   Simple RAG: {'✅ Available' if simple_available else '❌ Not Available'}")
    print(f"   Complete System: {'✅ Working' if system_test_ok else '❌ Issues'}")
    
    if chromadb_available:
        rag_mode = "ChromaDB (Production Ready)"
    elif simple_available:
        rag_mode = "Simple RAG (Development Mode)"
    else:
        rag_mode = "Fallback Responses Only"
    
    print(f"\n🎯 RAG Mode: {rag_mode}")
    
    print("\n📋 Next Steps:")
    if not system_test_ok:
        print("❌ 1. Fix system issues above before proceeding")
    else:
        print("✅ 1. Edit .env file and add your OpenAI API key")
        print("✅ 2. Test local server: uvicorn main:app --reload")
        print("✅ 3. Test endpoints: http://localhost:8000/rag-status")
        print("✅ 4. Deploy to Railway with same requirements.txt")
    
    print("\n🚀 Railway Deployment:")
    print("   - Same requirements.txt will work on Railway (Linux)")
    print("   - ChromaDB will likely install successfully on Railway")
    print("   - Your code auto-detects and uses best available RAG system")
    
    if chromadb_available:
        print("\n🎉 SETUP COMPLETE - Production Ready!")
    elif simple_available:
        print("\n🎉 SETUP COMPLETE - Development Ready!")
        print("   (ChromaDB will work on Railway deployment)")
    else:
        print("\n⚠️  SETUP PARTIAL - Basic functionality available")

if __name__ == "__main__":
    main()