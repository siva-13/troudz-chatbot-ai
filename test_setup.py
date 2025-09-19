# Quick Test Script
import os

print("🧪 RAG Application Quick Test")
print("=" * 30)

required_files = [
    'requirements.txt', 'document_processor.py', 'vector_store.py', 
    'dynamodb_manager.py', 'conversation_engine.py', 'main_app.py', 
    'fastapi_server.py', 'README.md'
]

print("📁 File Check:")
all_present = True
for file in required_files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"✅ {file} ({size:,} bytes)")
    else:
        print(f"❌ {file} - MISSING")
        all_present = False

if all_present:
    print("\n🎉 All files present! RAG application ready to use.")
    print("\n📖 Next Steps:")
    print("1. pip install -r requirements.txt")
    print("2. Set OPENAI_API_KEY environment variable")  
    print("3. Configure AWS credentials")
    print("4. Run: python fastapi_server.py")
else:
    print("\n⚠️  Some files are missing. Please ensure all files are extracted.")
