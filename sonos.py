import os
import socket
import threading
import logging
import time
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
import soco

##
##  Plays a .wav audio file to any sonos speaker with the IP address alone (must be on same local network as SONOS)
##

# Change these parameters as needed
SONOS_IP = "192.168.1.27"
HTTP_PORT = 8000
WAV_FILENAME = "recording.wav"  # Make sure this file exists in the current directory

# Configure logging to display detailed diagnostics of SOAP comms
logging.basicConfig(level=logging.ERROR, format='%(asctime)s [%(levelname)s] %(message)s')

class SilentHTTPServer(HTTPServer):
    """
    A custom HTTPServer that ignores ConnectionResetError exceptions during request handling (for SOAP).
    """
    def handle_error(self, request, client_address):
        exc_type, exc_value, tb = sys.exc_info()
        # If the error is a ConnectionResetError, just log it at debug level and ignore it.
        if isinstance(exc_value, ConnectionResetError):
            logging.debug("Ignoring ConnectionResetError from client %s", client_address)
        else:
            super().handle_error(request, client_address)

def get_wav_duration(file_path):
    import wave
    with wave.open(file_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
    return duration

def send_to_sonos(file_path, sonos_ip=None):
    """
    Serves the specified WAV file via a temporary HTTP server and instructs
    the Sonos speaker to play it, using the specified speaker IP address. 
    This is the imported function for the front-end.
    """
    #print(sonos_ip)
    if not os.path.isfile(file_path):
        logging.error("File '%s' not found.", file_path)
        return

    if sonos_ip is None:
        sonos_ip = SONOS_IP  # Use the default value if no IP is provided in the call

    # Serve the directory containing the file
    directory = os.path.dirname(os.path.abspath(file_path))
    original_dir = os.getcwd()
    os.chdir(directory)
    
    server_address = ('', HTTP_PORT)
    httpd = SilentHTTPServer(server_address, MyHTTPRequestHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    logging.info("HTTP server started in directory '%s' on port %d", directory, HTTP_PORT)

    time.sleep(1)

    local_ip = get_local_ip()
    file_name = os.path.basename(file_path)
    file_url = f"http://{local_ip}:{HTTP_PORT}/{file_name}"
    logging.info("Sending file URL to Sonos: %s", file_url)

    try:
        sonos = soco.SoCo(sonos_ip)
        sonos.play_uri(file_url)
        logging.info("play_uri command sent to Sonos speaker at %s", sonos_ip)
    except Exception as e:
        logging.exception("Error sending play command to Sonos:")
        httpd.shutdown()
        server_thread.join()
        os.chdir(original_dir)
        return

    duration = get_wav_duration(file_path)
    logging.info("Waiting for playback to complete (~%.2f seconds)...", duration)
    time.sleep(duration + 1)

    try:
        sonos.stop()
        logging.info("Stopped Sonos playback.")
    except Exception as e:
        logging.exception("Error stopping Sonos playback:")

    httpd.shutdown()
    server_thread.join()
    os.chdir(original_dir)
    logging.info("HTTP server shut down; control returning to the application.")

def get_local_ip():
    """
    Determines the local IP address of this machine as seen by devices on the network.
    This address will be used to build the URL of the specified file that the Sonos speaker can access.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Use an external address; note that no data is actually sent.
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        logging.debug("Determined local IP: %s", ip)
    except Exception as e:
        logging.exception("Could not determine local IP address:")
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

class MyHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        logging.info(format % args)

def start_http_server(port, directory):
    os.chdir(directory)
    server_address = ('', port)
    httpd = SilentHTTPServer(server_address, MyHTTPRequestHandler)
    logging.info("Starting HTTP server in directory '%s' on port %d", directory, port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logging.info("HTTP server interrupted by user. Shutting down.")
    httpd.server_close()

def launch_http_server_in_thread(port, directory):
    """
    Launches the HTTP server in a daemon thread so it runs in the background.
    """
    server_thread = threading.Thread(target=start_http_server, args=(port, directory), daemon=True)
    server_thread.start()
    return server_thread

def main():
    logging.info("Starting Sonos audio test program.")

    # Verify the WAV file exists in the current directory.
    if not os.path.isfile(WAV_FILENAME):
        logging.error("WAV file '%s' not found in current directory: %s", WAV_FILENAME, os.getcwd())
        sys.exit(1)

    # Start an HTTP server to serve the current directory (and thus the WAV file).
    current_dir = os.getcwd()
    logging.debug("Current directory: %s", current_dir)
    server_thread = launch_http_server_in_thread(HTTP_PORT, current_dir)

    # Wait briefly to ensure the HTTP server is running.
    time.sleep(1)

    # Get the local IP so the Sonos speaker can access the hosted file.
    local_ip = get_local_ip()
    file_url = f"http://{local_ip}:{HTTP_PORT}/{WAV_FILENAME}"
    logging.info("WAV file URL: %s", file_url)

    # Attempt to connect to the Sonos speaker.
    try:
        logging.info("Attempting to connect to Sonos speaker at %s", SONOS_IP)
        sonos = soco.SoCo(SONOS_IP)
        info = sonos.get_speaker_info()  # Retrieve a dictionary with speaker details
        zone_name = info.get("zoneName", "Unknown Zone")  # Get zone name safely
        logging.info("Connected to Sonos speaker: Zone Name: %s", zone_name)
    except Exception as e:
        logging.exception("Error connecting to Sonos speaker at %s", SONOS_IP)
        sys.exit(1)

    # Instruct the Sonos speaker to play the WAV file from our HTTP server.
    try:
        logging.info("Sending play_uri command to Sonos with URL: %s", file_url)
        sonos.play_uri(file_url)
        time.sleep(2)  # Wait briefly for the command to take effect.
        transport_info = sonos.get_current_transport_info()
        logging.info("Transport Info: %s", transport_info)
        track_info = sonos.get_current_track_info()
        logging.info("Current Track Info: %s", track_info)
    except Exception as e:
        logging.exception("Error sending play command to Sonos:")
        sys.exit(1)

    logging.info("Playback command sent successfully. The Sonos speaker should now stream the WAV file.")

    # Keep the program running until interrupted so you can view further logs.
    #try:
    #    while True:
    #        time.sleep(1)
    #except KeyboardInterrupt:
    #    logging.info("Test program interrupted by user. Exiting.")

if __name__ == "__main__":
    main()