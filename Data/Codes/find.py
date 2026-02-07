import os

print("--- DIAGNOSTIC REPORT ---")
print(f"1. Python thinks 'Here' (CWD) is: {os.getcwd()}")

if os.path.exists("synthetic_music_fraud_data.csv"):
    print(f"2. SUCCESS: File found at: {os.path.abspath('synthetic_music_fraud_data.csv')}")
    print("   (Copy this path and paste it into your File Explorer)")
else:
    print("3. FAILURE: File not found in CWD.")
    print("   It likely saved to a different location or wasn't generated.")
    print("   Try running the generation script again, but check your terminal path first.")