import firebase_admin
from firebase_admin import credentials, firestore

# Path to your service account key file
cred = credentials.Certificate(r"./moa-groq-firebase-adminsdk-qo6gc-827484e874.json")

# Check if the app is already initialized
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
else:
    firebase_admin.initialize_app(cred, name='unique_app_name')

# Firestore database instance
db = firestore.client()
