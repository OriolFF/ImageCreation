"""
Simple web server for the Image Creator UI.
Generates config.js with the specified API port before serving.

Usage:
    python serve.py                     # Uses default API port 8000
    python serve.py --api-port 8050     # Uses API port 8050
    python serve.py -p 5500 --api-port 8050  # Custom web port + API port
"""

import argparse
import os
import http.server
import socketserver

def generate_config(api_port: int, web_dir: str):
    """Generate config.js with the API base URL."""
    config_content = f'// Auto-generated config - do not edit manually\nconst API_PORT = {api_port};\nconst API_BASE = `http://127.0.0.1:${{{api_port}}}`;\n'
    config_path = os.path.join(web_dir, "config.js")
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"[Config] Generated {config_path} with API_PORT={api_port}")

def main():
    parser = argparse.ArgumentParser(description="Serve the Image Creator web UI")
    parser.add_argument("-p", "--port", type=int, default=5500, help="Web server port (default: 5500)")
    parser.add_argument("--api-port", type=int, default=8000, help="API server port (default: 8000)")
    parser.add_argument("-d", "--directory", type=str, default="web", help="Directory to serve (default: web)")
    args = parser.parse_args()

    # Generate config.js with the API port
    generate_config(args.api_port, args.directory)

    # Change to the web directory and serve
    os.chdir(args.directory)
    
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        print(f"[Server] Serving '{args.directory}' at http://localhost:{args.port}")
        print(f"[Server] API configured at http://127.0.0.1:{args.api_port}")
        print("[Server] Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[Server] Shutting down...")

if __name__ == "__main__":
    main()
