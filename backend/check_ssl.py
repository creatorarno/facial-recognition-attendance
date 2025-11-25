import ssl
import sys

print(f"Python: {sys.version}")
print(f"OpenSSL: {ssl.OPENSSL_VERSION}")
print(f"Default Verify Paths: {ssl.get_default_verify_paths()}")
try:
    ctx = ssl.create_default_context()
    print(f"Default Protocol: {ctx.protocol}")
except Exception as e:
    print(f"Error creating default context: {e}")
