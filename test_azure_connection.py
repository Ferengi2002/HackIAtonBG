"""
Test rapido de conexion hacia Azure AI Foundry (gpt-4o)
Verifica que las variables de entorno en .env funcionen correctamente.
"""

import os
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o").strip()

print("=" * 70)
print("VERIFICACION DE CONEXION CON AZURE AI FOUNDRY (gpt-4o)")
print("=" * 70)
print(f"Endpoint:   {endpoint}")
print(f"Deployment: {deployment}")
masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "(No configurada)"
print(f"API Key:    {masked_key}\n")

if not api_key or "PEGA_AQUI" in api_key or "tu_azure" in api_key:
    print("[AVISO] Todavia no has pegado tu Clave de API real en el archivo .env.")
    print("Abre el archivo '.env' y reemplaza 'PEGA_AQUI_TU_API_KEY' con tu clave copiada de Azure.")
    sys.exit(0)

try:
    from openai import OpenAI

    base_url = endpoint.rstrip("/")
    if base_url.endswith("/responses"):
        base_url = base_url[:-10]
    if not base_url.endswith("/openai/v1") and "/api/projects/" not in base_url:
        base_url = f"{base_url}/openai/v1"

    client = OpenAI(base_url=base_url, api_key=api_key)

    print("Enviando consulta de prueba a Azure AI Foundry...")
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "Eres el copiloto de Banco Guayaquil. Responde en una sola linea."},
            {"role": "user", "content": "Hola, confirma que estas activo."},
        ],
        max_tokens=50,
        temperature=0.2,
    )

    respuesta_texto = response.choices[0].message.content
    print("\n[EXITO] Conexion establecida correctamente con Azure AI Foundry!")
    print(f"Respuesta de {deployment}: \"{respuesta_texto}\"")
    print("=" * 70)

except Exception as exc:
    err_str = str(exc)
    if "still being initialized" in err_str:
        print("\n[EN PROCESO EN AZURE] ¡Tu API Key y Endpoint son 100% CORRECTOS!")
        print("Azure respondio: 'Instant inference quota is still being initialized for this subscription.'")
        print("Esto es normal cuando el recurso o deployment en Azure AI Foundry se acaba de crear.")
        print("Microsoft tarda entre 2 y 5 minutos en propagar la cuota regional. Vuelve a probar en unos instantes.")
    else:
        print(f"\n[ERROR al conectar con Azure]: {exc}")
        print("\nRevisa que:")
        print("1. La clave de API sea exactamente la de Key 1 o Key 2 en Azure AI Foundry.")
        print("2. El nombre del deployment ('gpt-4o') coincida con el nombre que le diste al implementarlo.")
    print("=" * 70)

