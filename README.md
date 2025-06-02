# 🧠 GenNarrate
A generative artificial intelligence project by @markuslw and @toropdahl for the INF-3993 course @ UiT

---
This project is a web application that allows users to provide an LLM with prompts or files, which the LLM uses in combination with RAG to provide a response. The application is built using Flask for the backend and inference, and React.js for the frontend.

## 📁 Structure

```
GenNarrate/
├── backend/ # Flask API
├── frontend/ # React.js interface
├── inference/ # Inference engine (Flask)
```

## 🔧 Setup
> [!NOTE]
>The project is developed on Linux Ubuntu 24.04.2 LTS. Depending on your OS, you may need to adapt the commands and packages.

### 🧑‍💻 Flask API
> [!TIP]
> Developed using Python 3.12.3 or higher.

1. navigate to the `backend/` directory
    ```bash
    $ cd backend/
    ```
2. create and activate a virtual environment
    ```bash
    $ python3 -m venv <name>
    $ source <name>/bin/activate
    ```
3. install the requirements
    ```bash
    $ pip install -r requirements.txt
    ```
4. run the server
    ```bash
    $ python3 main.py
    ```

### ⚛️ React.js
> [!TIP]
> The project was developed using Node.js v18.19.1, and npm v9.2.0.

If you do not have Node.js installed or are unsure,
check with
```bash
$ node -v
$ npm -v
```
which should return the version numbers if installed. In the event that you do not have Node.js installed, you can use the package manger by
```bash
$ sudo apt install nodejs npm
```

#### See below for setup instructuons

1. navigate to the `frontend/` directory
    ```bash
    $ cd frontend/
    ```
2. install the dependencies
    ```bash
    $ npm install
    ```
3. start the development server
    ```bash
    $ npm start
    ```

### 🧠 Inference Engine (Flask)
> [!WARNING]
> The inference engine is developed using Python 3.10.16. Not higher, not lower. Use the environment.yml file to create the conda environment with the correct Python version and packages.

Two shell scripts are provided to help run the inference engine on a remote server. `port_forward.sh` takes input and forwards a local port to a remote server and executes the Python script, while `kill_forwarding.sh` kills the process and the port forwarding.

The shell scripts provided in this directory assume that you have [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) installed with said Python version. To checked if you have miniconda installed, you can run the following command

```bash
$ ssh user@hostname
$ conda --version
```

Conda is needed for environments where permissions are limited, e.g. on a remote server with no sudo access.
Once installed, you will have to create a conda environment with the required packages. This can be done by running the following commands in the `inference/` directory once minisconda is installed

```bash
$ cd inference/
$ conda env create -f environment.yml # environment.yml file contains the required packages from apt and pip
$ conda activate inference # the name of the environment is inference as defined in the YML file
```

If you were to run into any issues with this environment, a `requirements.txt` file is also provided, which can be used to install the packages using `pip` while insdie the conda environment.

From here you can exit the remote server and run the shell scripts locally to forward the port and run the inference engine

```bash
$ chmod +x port_forward.sh
$ ./port_forward.sh <server_name> <local_port> <private_key> <work_dir>
```