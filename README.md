# 🧠 GenNarrate
A generative artificial intelligence project by @markuslw and @toropdahl for the INF-3993 course @ UiT

## 📁 Structure

```
GenNarrate/
├── backend/ # Django API
├── frontend/ # React.js interface
├── inference/ # Inference engine
```

## 🔧 Setup
> [!NOTE]
>The project is developed on Linux Ubuntu 24.04.2 LTS. Depending on your OS, you may need to adapt the commands and packages.

### 🧑‍💻 Django
> [!TIP]
> Developed using Python 3.12.3 or higher.

1. naviate to the `backend` directory
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
    $ python manage.py runserver
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

1. navigate to the `frontend` directory
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

### 🧠 Inference Engine
> [!WARNING]
> The inference engine is developed using Python 3.10.13. Not higher, not lower.

To install a lower python version, you could use miniconda which is used by the university as well.

More info coming soon.