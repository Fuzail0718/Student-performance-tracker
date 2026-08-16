#!/usr/bin/env python
"""
Main script to run the Student Performance Prediction Project
"""
import os
import sys
import subprocess
import time
import webbrowser
import platform

def print_banner():
    """Print project banner"""
    print("="*70)
    print(" 🎓 STUDENT PERFORMANCE PREDICTION PROJECT")
    print("="*70)
    print(" 📊 Machine Learning & Deep Learning for Student Success")
    print("="*70)

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'tensorflow', 
        'flask', 'streamlit', 'openpyxl', 'joblib'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\n📦 Installing missing packages...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
            print("✅ Packages installed successfully!")
            return True
        except:
            print("❌ Failed to install packages. Please run manually:")
            print("   pip install -r requirements.txt")
            return False
    else:
        print("✅ All required packages are installed")
        return True

def check_models():
    """Check if models exist"""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        print("📁 Created models directory")
        return False
    
    required_models = ['random_forest.pkl', 'grade_nn.h5']
    existing_models = []
    
    for model in required_models:
        if os.path.exists(os.path.join(models_dir, model)):
            existing_models.append(model)
    
    if len(existing_models) == len(required_models):
        print(f"✅ All models found: {', '.join(existing_models)}")
        return True
    else:
        print(f"⚠️  Models found: {', '.join(existing_models) if existing_models else 'None'}")
        print(f"   Required: {', '.join(required_models)}")
        return False

def train_models():
    """Train the models"""
    print("\n" + "="*70)
    print(" 🚀 TRAINING MODELS")
    print("="*70)
    
    # Check if training script exists
    if os.path.exists('train.py'):
        print("📚 Running training script...")
        try:
            result = subprocess.run(
                [sys.executable, 'train.py'],
                capture_output=False,
                check=True
            )
            print("✅ Models trained successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed with error: {e}")
            return False
    else:
        print("❌ Training script not found! (train.py)")
        return False

def open_browser(url, delay=2):
    """Open browser after delay"""
    def _open():
        time.sleep(delay)
        webbrowser.open(url)
    
    import threading
    threading.Thread(target=_open, daemon=True).start()

def start_services():
    """Start Flask API and Streamlit dashboard"""
    print("\n" + "="*70)
    print(" 🚀 STARTING SERVICES")
    print("="*70)
    
    processes = []
    ports = {'flask': 5000, 'streamlit': 8501}
    
    # Check if ports are available
    for port in ports.values():
        if is_port_in_use(port):
            print(f"⚠️  Port {port} is already in use!")
            response = input(f"   Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("❌ Service startup cancelled")
                return []
    
    # Start Flask API
    print("\n📡 Starting Flask API on http://localhost:5000")
    try:
        api_process = subprocess.Popen(
            [sys.executable, 'app/app.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(('Flask API', api_process, 5000))
        
        # Wait for API to start
        time.sleep(3)
        
        # Check if API started successfully
        if api_process.poll() is not None:
            stdout, stderr = api_process.communicate()
            print(f"❌ Flask API failed to start!")
            print(f"   Error: {stderr}")
            return []
        else:
            print("✅ Flask API started successfully!")
            
    except Exception as e:
        print(f"❌ Failed to start Flask API: {e}")
        return []
    
    # Start Streamlit Dashboard
    print("\n📊 Starting Streamlit Dashboard on http://localhost:8501")
    try:
        dashboard_process = subprocess.Popen(
            [sys.executable, '-m', 'streamlit', 'run', 'app/dashboard.py', '--server.port=8501'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(('Streamlit Dashboard', dashboard_process, 8501))
        
        # Wait for dashboard to start
        time.sleep(4)
        
        # Check if dashboard started successfully
        if dashboard_process.poll() is not None:
            stdout, stderr = dashboard_process.communicate()
            print(f"❌ Streamlit Dashboard failed to start!")
            print(f"   Error: {stderr}")
            return []
        else:
            print("✅ Streamlit Dashboard started successfully!")
            
    except Exception as e:
        print(f"❌ Failed to start Streamlit Dashboard: {e}")
        return []
    
    return processes

def is_port_in_use(port):
    """Check if a port is in use"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except socket.error:
            return True

def print_instructions(processes):
    """Print usage instructions"""
    print("\n" + "="*70)
    print(" ✅ SERVICES STARTED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📍 Access the services:")
    print("   📊 Streamlit Dashboard: http://localhost:8501")
    print("   🔮 Flask API: http://localhost:5000")
    print("   📝 API Documentation: http://localhost:5000/")
    print("   ❤️  Health Check: http://localhost:5000/health")
    print("   🔮 Prediction Form: http://localhost:5000/predict")
    
    print("\n💡 Sample API Call:")
    print("   curl -X POST http://localhost:5000/predict \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"Semester\":1,\"InternalMarks\":45,\"ExternalMarks\":35}'")
    
    print("\n📝 Available commands:")
    print("   - Train models: python train.py")
    print("   - Run all: python run_project.py")
    print("   - Stop services: Press Ctrl+C")
    
    print("\n🛑 Press Ctrl+C to stop all services...")

def cleanup_processes(processes):
    """Cleanup all running processes"""
    print("\n🛑 Stopping services...")
    for name, process, port in processes:
        try:
            process.terminate()
            print(f"✅ {name} stopped")
        except:
            pass
    
    # Force kill any remaining processes
    time.sleep(1)
    for name, process, port in processes:
        try:
            if process.poll() is None:
                process.kill()
                print(f"✅ {name} force killed")
        except:
            pass
    
    print("✅ All services stopped.")

def main():
    """Main execution"""
    print_banner()
    
    # Check if app directory exists
    if not os.path.exists('app'):
        print("\n❌ Error: 'app' directory not found!")
        print("Please make sure you're in the project root directory.")
        print("Current directory:", os.getcwd())
        sys.exit(1)
    
    # Check requirements
    print("\n📦 Checking requirements...")
    if not check_requirements():
        print("❌ Please install requirements and try again.")
        sys.exit(1)
    
    # Check data
    data_path = 'data/raw/dst data.xlsx'
    if not os.path.exists(data_path):
        print(f"\n❌ Data file not found: {data_path}")
        print("\nPlease make sure:")
        print("1. Your dataset is named 'dst data.xlsx'")
        print("2. It's located in: data/raw/")
        print("3. Current files in data/raw/:", 
              os.listdir('data/raw') if os.path.exists('data/raw') else "Directory not found")
        sys.exit(1)
    else:
        print(f"✅ Data found: {data_path}")
    
    # Check and train models
    print("\n🤖 Checking models...")
    models_ready = check_models()
    
    if not models_ready:
        print("\n📚 Models need to be trained.")
        response = input("   Train models now? (y/n): ")
        if response.lower() == 'y':
            if not train_models():
                print("❌ Model training failed. Please train manually.")
                sys.exit(1)
        else:
            print("❌ Models are required to run the project.")
            print("   Run 'python train.py' to train models.")
            sys.exit(1)
    
    # Start services
    processes = start_services()
    
    if not processes:
        print("\n❌ Failed to start services. Please check the logs.")
        sys.exit(1)
    
    # Open browsers automatically
    print("\n🌐 Opening browsers...")
    open_browser('http://localhost:8501', 1)  # Dashboard
    open_browser('http://localhost:5000', 3)  # API
    
    # Print instructions
    print_instructions(processes)
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
            
            # Check if any process died
            for name, process, port in processes:
                if process.poll() is not None:
                    print(f"\n⚠️  {name} process died!")
                    print(f"   Port {port} may still be in use.")
                    print("   You may need to restart the service.")
                    
                    # Try to restart the process
                    if name == 'Flask API':
                        print(f"🔄 Attempting to restart {name}...")
                        try:
                            new_process = subprocess.Popen(
                                [sys.executable, 'app/app.py'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True
                            )
                            # Update the process in the list
                            for i, (n, p, port) in enumerate(processes):
                                if n == name:
                                    processes[i] = (name, new_process, port)
                                    break
                            print(f"✅ {name} restarted")
                        except:
                            print(f"❌ Failed to restart {name}")
                    
                    elif name == 'Streamlit Dashboard':
                        print(f"🔄 Attempting to restart {name}...")
                        try:
                            new_process = subprocess.Popen(
                                [sys.executable, '-m', 'streamlit', 'run', 'app/dashboard.py', '--server.port=8501'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True
                            )
                            # Update the process in the list
                            for i, (n, p, port) in enumerate(processes):
                                if n == name:
                                    processes[i] = (name, new_process, port)
                                    break
                            print(f"✅ {name} restarted")
                        except:
                            print(f"❌ Failed to restart {name}")
                    
    except KeyboardInterrupt:
        cleanup_processes(processes)
        print("\n👋 Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()