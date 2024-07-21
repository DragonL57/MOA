import firebase_admin
from firebase_admin import credentials, firestore

# Use raw string or forward slashes for the file path to avoid unicode errors
cred = credentials.Certificate(r".\moa-groq-firebase-adminsdk-qo6gc-827484e874.json")
firebase_admin.initialize_app(cred)

db = firestore.client()
