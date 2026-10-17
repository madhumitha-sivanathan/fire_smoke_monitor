from pathlib import Path
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# CHANGE THIS to the actual location of your YOLOv10 model
MODEL_PATH = PROJECT_ROOT / "models" / "model.pt"

IMAGE_FOLDER = PROJECT_ROOT / "images"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

PRODUCTION_CONF_THRESHOLD = 0.25
DIAGNOSTIC_CONF_THRESHOLD = 0.001

IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(MODEL_PATH)


print("Model classes:")
print(model.names)

print("\nConfiguration:")
print(f"Production confidence threshold : {PRODUCTION_CONF_THRESHOLD}")
print(f"Diagnostic confidence threshold : {DIAGNOSTIC_CONF_THRESHOLD}")
print(f"IoU threshold                   : {IOU_THRESHOLD}")
print(f"Image size                      : {IMAGE_SIZE}")


# ============================================================
# DETERMINE EXPECTED CATEGORY
# ============================================================

def get_expected_category(image_path):
    """
    Determine expected category from filename.

    fire_01.jpg      -> fire
    smoke_01.jpg     -> smoke
    normal_01.jpg    -> none
    extreme_fire.jpg -> fire
    """

    name = image_path.stem.lower()

    if "fire" in name:
        return "fire"

    if "smoke" in name:
        return "smoke"

    if "normal" in name:
        return "none"

    return "unknown"


# ============================================================
# EXTRACT BEST CLASS CONFIDENCES
# ============================================================

def get_best_confidences(result):
    """
    Find the highest Fire and Smoke confidence
    from a YOLO result.
    """

    best_fire = 0.0
    best_smoke = 0.0

    if result.boxes is None or len(result.boxes) == 0:
        return best_fire, best_smoke

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = result.names[class_id].lower()

        if class_name == "fire":
            best_fire = max(best_fire, confidence)

        elif class_name == "smoke":
            best_smoke = max(best_smoke, confidence)

    return best_fire, best_smoke


# ============================================================
# GET PRODUCTION PREDICTION
# ============================================================

def get_production_prediction(result):
    """
    Reproduce the current EdgeGateway selection logic:

        best_box = max(result.boxes, key=confidence)

    Returns:
        class_name, confidence
    """

    if result.boxes is None or len(result.boxes) == 0:
        return None, 0.0

    best_box = max(
        result.boxes,
        key=lambda box: float(box.conf[0])
    )

    class_id = int(best_box.cls[0])
    confidence = float(best_box.conf[0])

    class_name = result.names[class_id].lower()

    return class_name, confidence


# ============================================================
# GET ALL PRODUCTION DETECTIONS
# ============================================================

def get_retained_detections(result):
    """
    Return all detections that survive the production
    confidence threshold.

    This is mainly for diagnosis.
    """

    detections = []

    if result.boxes is None or len(result.boxes) == 0:
        return detections

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = result.names[class_id]

        if confidence >= PRODUCTION_CONF_THRESHOLD:

            detections.append(
                (class_name, confidence)
            )

    detections.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return detections


# ============================================================
# FIND IMAGES
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

image_paths = sorted(
    [
        path
        for path in IMAGE_FOLDER.iterdir()
        if path.suffix.lower() in image_extensions
    ]
)

print(f"\nFound {len(image_paths)} image(s).")


# ============================================================
# SUMMARY COUNTERS
# ============================================================

total_images = 0

correct_predictions = 0
wrong_predictions = 0
no_predictions = 0

fire_images = 0
smoke_images = 0
normal_images = 0

fire_correct = 0
smoke_correct = 0
normal_correct = 0


# ============================================================
# TEST EACH IMAGE
# ============================================================

for image_path in image_paths:

    total_images += 1

    expected = get_expected_category(image_path)

    if expected == "fire":
        fire_images += 1

    elif expected == "smoke":
        smoke_images += 1

    elif expected == "none":
        normal_images += 1

    print("\n" + "=" * 70)
    print(f"IMAGE: {image_path.name}")
    print("=" * 70)

    print(f"Expected category : {expected.upper()}")

    # --------------------------------------------------------
    # Diagnostic inference
    # --------------------------------------------------------

    diagnostic_results = model.predict(
        source=str(image_path),
        conf=DIAGNOSTIC_CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        verbose=False
    )

    diagnostic_result = diagnostic_results[0]

    best_fire, best_smoke = get_best_confidences(
        diagnostic_result
    )

    print("\nDiagnostic inference:")
    print(f"Best Fire confidence  : {best_fire:.4f}")
    print(f"Best Smoke confidence : {best_smoke:.4f}")

    # --------------------------------------------------------
    # Production inference
    # --------------------------------------------------------

    production_results = model.predict(
        source=str(image_path),
        conf=PRODUCTION_CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        verbose=False
    )

    production_result = production_results[0]

    predicted_class, predicted_confidence = (
        get_production_prediction(production_result)
    )

    retained = get_retained_detections(
        production_result
    )

    print("\nProduction inference:")

    if predicted_class is None:

        print("Prediction             : NONE")
        print("Confidence             : 0.0000")

        prediction_category = "none"

        no_predictions += 1

    else:

        print(
            f"Prediction             : "
            f"{predicted_class.upper()}"
        )

        print(
            f"Confidence             : "
            f"{predicted_confidence:.4f}"
        )

        prediction_category = predicted_class

    # --------------------------------------------------------
    # Retained detections
    # --------------------------------------------------------

    print("\nRetained detections:")

    if not retained:

        print("  None")

    else:

        for class_name, confidence in retained:

            print(
                f"  {class_name:<7} "
                f"{confidence:.4f}"
            )

    # --------------------------------------------------------
    # Evaluate prediction
    # --------------------------------------------------------

    if expected == "unknown":

        print("\nEvaluation             : UNKNOWN CATEGORY")

        continue

    if expected == prediction_category:

        print("Evaluation             : CORRECT")

        correct_predictions += 1

        if expected == "fire":
            fire_correct += 1

        elif expected == "smoke":
            smoke_correct += 1

        elif expected == "none":
            normal_correct += 1

    else:

        print("Evaluation             : WRONG")

        wrong_predictions += 1

        # Extra diagnosis
        if expected == "fire":

            if prediction_category == "smoke":

                print(
                    "Diagnosis              : "
                    "WRONG CLASS - FIRE → SMOKE"
                )

            elif prediction_category == "none":

                print(
                    "Diagnosis              : "
                    "FALSE NEGATIVE - FIRE MISSED"
                )

        elif expected == "smoke":

            if prediction_category == "fire":

                print(
                    "Diagnosis              : "
                    "WRONG CLASS - SMOKE → FIRE"
                )

            elif prediction_category == "none":

                print(
                    "Diagnosis              : "
                    "FALSE NEGATIVE - SMOKE MISSED"
                )

        elif expected == "none":

            if prediction_category == "fire":

                print(
                    "Diagnosis              : "
                    "FALSE POSITIVE - FIRE"
                )

            elif prediction_category == "smoke":

                print(
                    "Diagnosis              : "
                    "FALSE POSITIVE - SMOKE"
                )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL MODEL BEHAVIOUR SUMMARY")
print("=" * 70)

print(f"\nTotal images tested : {total_images}")

print("\nImages by category:")
print(f"  Fire   : {fire_images}")
print(f"  Smoke  : {smoke_images}")
print(f"  Normal : {normal_images}")

print("\nPrediction results:")
print(f"  Correct predictions : {correct_predictions}")
print(f"  Wrong predictions   : {wrong_predictions}")
print(f"  No prediction       : {no_predictions}")

if total_images > 0:

    accuracy = (
        correct_predictions / total_images
    ) * 100

    print(
        f"\nOverall accuracy     : "
        f"{accuracy:.2f}%"
    )

if fire_images > 0:

    fire_accuracy = (
        fire_correct / fire_images
    ) * 100

    print(
        f"Fire recognition     : "
        f"{fire_accuracy:.2f}%"
    )

if smoke_images > 0:

    smoke_accuracy = (
        smoke_correct / smoke_images
    ) * 100

    print(
        f"Smoke recognition    : "
        f"{smoke_accuracy:.2f}%"
    )

if normal_images > 0:

    normal_accuracy = (
        normal_correct / normal_images
    ) * 100

    print(
        f"Normal recognition   : "
        f"{normal_accuracy:.2f}%"
    )

print("\n" + "=" * 70)
print("Diagnostic complete.")
print("=" * 70)