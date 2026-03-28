# Live YOLO26m Detection

Dieses Beispiel zeigt ein Live-Kamerabild mit OpenCV und fuehrt darauf YOLO-Detektion mit einem `yolo26m.pt`-Modell aus.
Die Ultralytics-Synchronisierung wird im Skript per `settings.update({"sync": False})` deaktiviert.

Zusätzlich gibt es `live_yoloe.py` fuer YOLOE mit frei definierbaren Text-Prompts.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## TensorFlow.js Export

Der normale Projekt-Setup installiert bewusst keine TensorFlow.js-Export-Toolchain, damit `pip install -r requirements.txt`
fuer Live-Inferenz und Training stabil bleibt.

Fuer den Export nach TensorFlow.js eine separate Export-Umgebung verwenden, idealerweise mit Python 3.12:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv_tfjs
source .venv_tfjs/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-tfjs-export.txt
python export_yolo_tfjs.py --model runs/detect/runs/train/yolo26m-custom/weights/best.pt
```

Hinweis:
Die Python-TF.js-Toolchain haengt an zusaetzlichen TensorFlow-/onnx2tf-Abhaengigkeiten und ist nicht fuer jede
Python-Version gleich gut aufloesbar. Deshalb ist sie absichtlich aus `requirements.txt` ausgelagert.

## Start

```bash
python3 live_yolo26m.py --model yolo26m.pt
```

YOLOE-Beispiel:

```bash
python3 live_yoloe.py --model yoloe-11s-seg.pt --prompt person --prompt bus
```

Alternativ koennen mehrere Klassen auch komma-separiert angegeben werden:

```bash
python3 live_yoloe.py --model yoloe-11s-seg.pt --prompt "person,bus,dog"
```

Optionale Parameter:

- `--mode predict|track`
- `--tracker bytetrack.yaml`
- `--track-history-seconds 3.0`
- `--camera-index 0`
- `--conf 0.25`
- `--imgsz 640`

Beenden mit `q` oder `Esc`.

Tracking-Beispiel:

```bash
python3 live_yolo26m.py --model yolo26m.pt --mode track --tracker bytetrack.yaml
```

Dabei wird im Live-Bild die Spur der letzten 3 Sekunden pro Track eingezeichnet.

Hinweis zu YOLOE:
Falls das Initialisieren des Text-Prompts fehlschlaegt, fehlen meist YOLOE-spezifische Textmodell-Abhaengigkeiten. In aktuellen Ultralytics-Setups wird dafuer haeufig das GitHub-Paket `clip` benoetigt, nicht das gleichnamige PyPI-Paket.

# YOLO-E:

```bash
python live_yoloe.py --model yoloe-11s-seg.pt --camera-index 1 --mode track --prompt "metal ball"
```

![yolo e demo](images/yoloe_demo.png)


# Fine train with one class using YOLO


## Step by Step:
1. make pictures
2. Install [label-studio](https://labelstud.io) to create label dataset  
   ```bash
   brew install python@3.13
   /opt/homebrew/bin/python3.13 -m venv .venv
   . .venv/bin/activate
   python -m pip install -U pip setuptools wheel
   python -m pip install label-studio
   label-studio
   ```
3. Create new account, create new project.
4. Labeling Setup: `Object Detection with Bounding Boxes`
5. export as `YOLO with Images`
6. `python train_yolo26m.py --device mps` from this repo: https://github.com/pbotte/scattering-cam

### Hints:
- Documentation: https://docs.ultralytics.com/datasets/detect/?utm_source=chatgpt.com
- Alternatively: https://github.com/cvat-ai/cvat

## Inference
Using this repo: https://github.com/pbotte/scattering-cam
```bash
python live_yolo26m.py --model runs/detect/runs/train/yolo26m-custom/weights/best.pt --camera-index 1 --mode track
```

![demo own model](images/ownmodel_demo.png)

## TensorFlow.js Webcam Demo

Das trainierte Modell kann auch als TensorFlow.js-Modell im Browser genutzt werden.

### Export

Der Export erzeugt das Browser-Modell unter:

```text
runs/detect/runs/train/yolo26m-*/weights/best_web_model
```

Export aus einer kompatiblen Python-Umgebung:

```bash
. .venv_3.11/bin/activate
python export_yolo_tfjs.py --model runs/detect/runs/train/yolo26m-custom/weights/best.pt
```

### Start der HTML-Demo

Die Demo-Datei liegt hier:

```text
tfjs_webcam_demo.html
```

Die Seite nicht per `file://` oeffnen, sondern ueber einen lokalen Webserver:

```bash
python3 -m http.server 8000
```

Dann im Browser:

```text
http://localhost:8000/tfjs_webcam_demo.html
```

### Firefox-Hinweis

Firefox konnte in diesem Setup TensorFlow.js nicht immer direkt vom CDN laden.
Deshalb versucht die Demo zuerst eine lokale Datei:

```text
vendor/tf.min.js
```

Falls die Datei noch fehlt:

```bash
mkdir -p vendor
curl -L https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js -o vendor/tf.min.js
```

### Bedienung

- `Webcam Starten` startet Kamera und Inferenz
- `Stopp` beendet Kamera und Inferenz
- Im Dropdown `Kamera` kann zwischen mehreren Kameras gewechselt werden
- Mit `Confidence` laesst sich der Anzeige-Schwellwert anpassen
