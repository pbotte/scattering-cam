# PROTOKOLL TFJS

## Endergebnisse

- TensorFlow.js-Modell erfolgreich exportiert unter `runs/detect/runs/train/yolo26m-custom/weights/best_web_model/`
  ```bash
  python export_yolo_tfjs.py --model runs/detect/runs/train/yolo26m-custom/weights/best.pt
  ```

- HTML-Webcam-Demo erfolgreich erstellt in `tfjs_webcam_demo.html`
  ```bash
  python3 -m http.server 8000
  ```
  ```text
  http://localhost:8000/tfjs_webcam_demo.html
  ```

- Demo funktioniert stabil mit sichtbarem `<video>` und Canvas-Overlay fuer die Erkennungsboxen

- Kameraauswahl, Start/Stopp, Confidence-Slider und FPS-Anzeige sind integriert

- Firefox funktioniert mit lokalem TensorFlow.js-Fallback `vendor/tf.min.js`
  ```bash
  mkdir -p vendor
  curl -L https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js -o vendor/tf.min.js
  ```

- Auf iOS ist fuer Kamera-Zugriff `https` oder `localhost` erforderlich
