# test_webhook.py
# Script de test LOCAL uniquement - Ne jamais commit ce fichier avec le vrai token en dur

from dotenv import load_dotenv
import os
import requests
import sys

def send_discord_notification(message="Test depuis mon script Python local"):
    """
    Envoie une notification Discord via webhook.
    À utiliser uniquement pour tester la connexion localement.
    """
    # Charge les variables du fichier .env
    load_dotenv()
    
    webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    
    if not webhook_url:
        print("❌ Erreur: DISCORD_WEBHOOK_URL non trouvé dans le fichier .env")
        print("Créez un fichier .env avec: DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...")
        sys.exit(1)
    
    # Masque l'URL pour l'affichage (sécurité)
    masked_url = webhook_url[:50] + "..." if len(webhook_url) > 50 else webhook_url
    print(f"Envoi vers: {masked_url}")
    
    payload = {
        "content": message,
        "username": "MLOps Local Test"
    }
    
    try:
        response = requests.post(webhook_url, json=payload)
        
        if response.status_code == 204:
            print("✅ Message envoyé avec succès!")
        else:
            print(f"❌ Erreur {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    # Test avec un message personnalisé
    send_discord_notification("🧪 Test de notification depuis mon environnement local")