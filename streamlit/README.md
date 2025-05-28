# GenNarrate - Streamlit Interface

This is the local user interface for the GenNarrate system. It communicates with a remote inference server via HTTP and provides a simple chat interface and PDF upload capability. The intended use is as a teaching assistant or book reader.

## Requirements

- Python 3.10 or newer
- A running remote inference server (e.g., using `port_forward.sh` to expose it on `localhost:5001`)
- The following Python packages:

  - `streamlit`
  - `requests`
  - `pymupdf` (for reading PDF files)

## Setup

1. Create a virtual environment (optional but recommended):

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    Example `requirements.txt`:

    ```
    streamlit
    requests
    pymupdf
    ```

3. Make sure the remote inference server is running and accessible at `http://localhost:5001`. If hosted on a remote server, you can use:

    ```bash
    ./port_forward.sh <user@host> 5001 <private_key_file> /path/to/main.py
    ```

## Running the Streamlit Interface

From the root directory (where `app.py` is located), run:

```bash
streamlit run app.py
