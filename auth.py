import firebase_admin
from firebase_admin import auth
from firebase_admin import firestore
from firebase_admin.exceptions import FirebaseError

db = firestore.client()

def create_user(email, password):
    try:
        user = auth.create_user(email=email, password=password)
        print(f'Successfully created new user: {user.uid}')
        return user
    except FirebaseError as e:
        print(f'Error creating new user: {e}')
        return None

def sign_in_user(email, password):
    try:
        user = auth.get_user_by_email(email)
        return user
    except FirebaseError as e:
        print(f'Error signing in user: {e}')
        return None

def store_conversation(user_id, conversation):
    try:
        doc_ref = db.collection('conversations').document(user_id)
        doc_ref.set({
            'conversations': conversation
        })
        print(f'Successfully stored conversation for user: {user_id}')
    except Exception as e:
        print(f'Error storing conversation: {e}')

def get_user_conversations(user_id):
    try:
        doc_ref = db.collection('conversations').document(user_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict().get('conversations', [])
        else:
            print('No conversation history found for user.')
            return []
    except Exception as e:
        print(f'Error retrieving conversations: {e}')
        return []
