#!/usr/bin/env python
"""
Simple launcher for model training from the project root
This script trains all models for the student performance prediction project
"""
import os
import sys
import subprocess
import time

def print_banner():
    """Print training banner"""
    print("="*70)
    print(" 🚀 STUDENT PERFORMANCE MODEL TRAINING")
    print("="*70)
    print(" 📊 Training: Random Forest, Neural Network & LSTM")
    print("="*70)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported!")
        print("   Please use Python 3.8 or higher")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'tensorflow', 
        'keras', 'matplotlib', 'seaborn', 'openpyxl', 'joblib'
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
        except subprocess.CalledProcessError:
            print("\n❌ Failed to install packages. Please run manually:")
            print("   pip install -r requirements.txt")
            return False
    else:
        print("✅ All required packages are installed")
        return True

def check_data():
    """Check if data file exists"""
    data_path = os.path.join('data', 'raw', 'dst data.xlsx')
    
    if not os.path.exists(data_path):
        print(f"\n❌ Data file not found: {data_path}")
        print("\nPlease make sure:")
        print("1. Your dataset is named 'dst data.xlsx'")
        print("2. It's located in: data/raw/")
        
        # Show what's in the data/raw directory
        if os.path.exists('data/raw'):
            files = os.listdir('data/raw')
            if files:
                print(f"   Files found in data/raw/: {', '.join(files)}")
            else:
                print("   data/raw/ directory is empty")
        else:
            print("   data/raw/ directory doesn't exist!")
            print("   Creating data/raw/ directory...")
            os.makedirs('data/raw', exist_ok=True)
        
        return False
    
    print(f"✅ Data found: {data_path}")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'models',
        'reports',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def run_training_script():
    """Run the actual training script"""
    print("\n" + "="*70)
    print(" 🔄 STARTING MODEL TRAINING")
    print("="*70)
    
    # Check if training script exists
    script_path = os.path.join('scripts', 'train_models.py')
    
    if not os.path.exists(script_path):
        print(f"❌ Training script not found: {script_path}")
        print("\nCreating training script...")
        
        # Create the scripts directory if it doesn't exist
        os.makedirs('scripts', exist_ok=True)
        
        print("❌ Please make sure 'scripts/train_models.py' exists")
        print("   You can create it manually or run:")
        print("   python -c \"import os; os.makedirs('scripts', exist_ok=True)\"")
        return False
    
    print(f"📚 Running training script: {script_path}")
    print("⏳ This may take a few minutes...")
    print("-"*70)
    
    try:
        # Run the training script
        start_time = time.time()
        
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            check=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("-"*70)
        print(f"✅ Training completed in {duration:.2f} seconds!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\n❌ Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def check_trained_models():
    """Check if models were created successfully"""
    models_dir = 'models'
    required_models = ['random_forest.pkl', 'grade_nn.h5']
    
    if not os.path.exists(models_dir):
        print("❌ Models directory not found")
        return False
    
    existing_models = []
    for model in required_models:
        model_path = os.path.join(models_dir, model)
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / 1024  # KB
            existing_models.append(f"{model} ({size:.1f} KB)")
    
    if len(existing_models) == len(required_models):
        print("\n✅ All models created successfully!")
        print(f"   📁 Models saved in: {models_dir}/")
        for model in existing_models:
            print(f"      - {model}")
        return True
    else:
        print(f"\n⚠️  Models created: {len(existing_models)}/{len(required_models)}")
        for model in existing_models:
            print(f"      - {model}")
        return False

def print_summary(success):
    """Print training summary"""
    print("\n" + "="*70)
    if success:
        print(" ✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\n📁 Models saved in: models/")
        print("   - random_forest.pkl")
        print("   - grade_nn.h5")
        print("   - lstm_model.h5 (if enough data)")
        
        print("\n📊 Reports saved in: reports/")
        print("   - feature_importance.csv")
        print("   - feature_importance.png")
        
        print("\n🚀 Next steps:")
        print("   1. Run the complete project:")
        print("      python run_project.py")
        print("   2. Or start services individually:")
        print("      python app/app.py          # Flask API")
        print("      streamlit run app/dashboard.py  # Dashboard")
    else:
        print(" ❌ MODEL TRAINING FAILED")
        print("="*70)
        print("\n🔧 Troubleshooting tips:")
        print("   1. Check if data file exists in: data/raw/dst data.xlsx")
        print("   2. Install requirements: pip install -r requirements.txt")
        print("   3. Check Python version: python --version")
        print("   4. Check error messages above")
    
    print("="*70)

def main():
    """Main execution"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Check requirements
    print("\n📦 Checking requirements...")
    if not check_requirements():
        print("\n❌ Please install requirements and try again.")
        sys.exit(1)
    
    # Check data
    print("\n📊 Checking data...")
    if not check_data():
        print("\n❌ Please place your dataset in data/raw/dst data.xlsx")
        sys.exit(1)
    
    # Run training
    success = run_training_script()
    
    # Check trained models
    if success:
        models_ok = check_trained_models()
        success = success and models_ok
    
    # Print summary
    print_summary(success)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()